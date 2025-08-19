import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SubsetRandomSampler
import optuna
import argparse
import pprint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (HYPERPARAM_DB_URI, 
                     GLOBAL_RANDOM_STATE,
                     REPO_DIR
                     )
from .models import FeedForwardNN
from .dataset import TabularDataset, load_data


# Class type of the custom dataset desired
DATASET_CLASS_TYPE = TabularDataset

# Directory to save tuning trials dataframe
STUDIES_DIR = REPO_DIR / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)


# --- 3. Define the Optuna Objective Function ---
def objective(trial: optuna.Trial,
              gpu_id: int,
              dataset: torch.utils.data.Dataset,
              num_folds: int,
              validation_split_ratio: float | None = None
    ) -> float:
    
    """
    The objective function for Optuna to optimize.
    A 'trial' represents a single run with a specific set of hyperparameters.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        gpu_id (int): The id of the NVIDIA CUDA-enabled GPU.
        dataset (torch.utils.data.Dataset): The training dataset for hyperparameter tuning.
        num_folds (int): The number of folds. If > 1, performs k-fold CV.
                         If = 1, performs a single train/validation split
                         and uses the `validation_split_ratio`.
        validation_split_ratio (float): The ratio of the dataset to be used for the
                                        validation set ONLY in a single split scenario
                                        (i.e. when `num_folds=1`)        
    """
    if num_folds == 1:
        assert validation_split_ratio is not None, \
            "Must give value for `validation_split_ratio` when `num_folds` is 1"

    # Setting device for this specific trial
    device = torch.device(f"cuda:{gpu_id}")

    # List of dataset indices to use for splitting
    dataset_indices = list(range(len(dataset)))
    
    # Determine the splitting strategy based on num_folds
    if num_folds > 1:
        kfold = KFold(n_splits=num_folds,
                      shuffle=True,
                      random_state=GLOBAL_RANDOM_STATE
        )
        splits = kfold.split(dataset_indices)
    else:
        train_indices, val_indices = train_test_split(
            dataset_indices,
            test_size=validation_split_ratio,
            random_state=GLOBAL_RANDOM_STATE
        )
        # Wrap in a list to use the same loop structure as k-fold
        splits = [(train_indices, val_indices)]


    # --- Hyperparameter Search Space ---
    n_layers = trial.suggest_int('n_layers', 1, 5) # Number of hidden layers
    hidden_units = tuple(
        trial.suggest_int(f"n_units_l{i}", 16, 256) for i in range(n_layers)
    )
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    epochs = trial.suggest_int('epochs', 100, 300)
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
    penalty_weight = trial.suggest_float('penalty_weight', 1e-4, 1e2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)


    # List for all (possibly only one) validation accuracies
    validation_accuracies = []
    # Loop through the splits
    for fold, (train_ids, val_ids) in enumerate(splits):

        # Creating a generator and setting its seed as the fold 
        # number so as to ensure that the random state of the 
        # generator starts from the beginning every fold and the 
        # state of producing pseudo-random numbers doesn't carry  
        # on (e.g. from fold 0 to fold 1)
        g = torch.Generator().manual_seed(fold)

        # Create data samplers and loaders for the current fold
        train_sampler = SubsetRandomSampler(train_ids, generator=g)
        val_sampler = SubsetRandomSampler(val_ids, generator=g)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=2
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=2
        )

        # --- Model, Optimizer, and Loss ---
        first_X, first_y = dataset[0]
        input_size = first_X.shape[0] if first_X.dim() > 0 else 1
        output_size = first_y.shape[0] if first_y.dim() > 0 else 1
        model = FeedForwardNN(
            input_features=input_size,
            hidden_units=hidden_units,
            output_features=output_size,
            dropout_rate=dropout_rate
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                # Calculate primary loss (i.e. w/o penalty loss)
                loss = criterion(outputs, batch_y)

                penalty_loss = 0
                for param in model.parameters():
                    if param.requires_grad:
                        # `param` is a tensor of weights 
                        # (can be a tensor of bias terms as well)
                        penalty_loss += \
                        ( l1_ratio * torch.sum(torch.abs(param)) ) \
                        + ( ((1-l1_ratio) / 2) * torch.sum(torch.square(param)) )

                loss += penalty_weight * penalty_loss

                loss.backward()
                optimizer.step() 
            # --- Finished an epoch of training        
        
        # --- Finished all epochs

        model.eval()
        fold_val_num_correct_list = [] # list of num correct preds for each batch
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                batch_y_val_pred_logits = model(batch_X_val).squeeze()
                batch_X_val_pred = (torch.sigmoid(batch_y_val_pred_logits) > 0.5).float()
                batch_num_correct = (batch_X_val_pred == batch_y_val).sum().item()
                fold_val_num_correct_list.append(batch_num_correct)

        # The total correct predictions divided by size of validation fold 
        fold_val_acc = sum(fold_val_num_correct_list) / len(val_ids)

        # --- Record this fold's validation accuracy
        validation_accuracies.append(fold_val_acc)
    # --- End of looping through splits

    # still calling it avg even when `num_folds==1`
    avg_validation_acc = sum(validation_accuracies) / len(validation_accuracies)
    
    return avg_validation_acc



def run_optimization(gpu_id: int,
                     dataset: torch.utils.data.Dataset,
                     study_name: str,
                     storage_name: str,
                     num_trials: int,
                     num_folds: int,
                     validation_split_ratio: float | None = None
    ):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    objective_fn = lambda trial: objective(trial, gpu_id, dataset, num_folds, validation_split_ratio)
    study.optimize(objective_fn, n_trials=num_trials)



def main():
    print("Start tuning script.")

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Optuna PyTorch Optimization Script")
    parser.add_argument("--n-gpus", type=int, required=True,
                        help="Number of GPUs on machine.")
    
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to data file.")
    
    parser.add_argument("--study-name", type=str, required=True, 
                        help="Name of the Optuna study.")
    
    parser.add_argument("--total-trials", type=int, required=True, 
                        help="Total number of trials desired for tuning. Will be split among n-gpus.")
    
    parser.add_argument("--num-folds", type=int, required=True, 
                        help="Number of folds to do in k-fold CV. Put 1 if you want single train/validation split.")
    
    parser.add_argument("--scale-features", action="store_true", required=False,
                        help="Include flag if you want to scale X variables.")
    
    parser.add_argument("--scale-targets", action="store_true", required=False,
                        help="Include flag if you want to scale y variables.")
    
    parser.add_argument("--train-split-ratio",type=float, required=True,
                        help="Proportion of the entire dataset to use as training.")    
    
    parser.add_argument("--validation-split-ratio",type=float, required=False,
                        help="Proportion of the training data to use as validation if doing a single train/validation split.")
    
    args = parser.parse_args()

    
    # Directory where all info about this study will be saved 
    # e.g. hyperparameters at each trial, 
    # epoch statistics of final model
    STUDY_OUTPUT_DIR = STUDIES_DIR / args.study_name
    STUDY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optuna Study args
    STUDY_ARGS = {
        "study_name": args.study_name,
        "storage": HYPERPARAM_DB_URI,
        "direction": "maximize",
        "load_if_exists": True,
        "sampler": None
    }

    mp.set_start_method('fork', force=True)
    
    # --- Load data ONCE in the main process ---
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor \
     = load_data(args.data_path,
                 args.scale_features,
                 args.scale_targets,
                 args.train_split_ratio
    )

    # --- Move tensors to shared memory ---
    X_train_tensor.share_memory_()
    y_train_tensor.share_memory_()
    print("\nData loaded and moved to shared memory.")

    hyperparam_train_dataset = DATASET_CLASS_TYPE(X_train_tensor, y_train_tensor)
    print("\nCreated hyperparameter torch Dataset object")

    # --- Creating Optuna study
    study = optuna.create_study(
        **STUDY_ARGS
    )
    print("\nCreated Optuna study")

    # --- Spawning of child processes ---
    num_gpus = args.n_gpus
    n_trials_gpu = args.total_trials // num_gpus
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=run_optimization, 
            args=(gpu_id,
                  hyperparam_train_dataset,
                  STUDY_ARGS["study_name"],
                  STUDY_ARGS["storage"],
                  n_trials_gpu,
                  args.num_folds,
                  args.validation_split_ratio
            )
        )
        p.start()
        processes.append(p)

    print(f"\nSpawned {num_gpus} processes.")

    for p in processes:
        p.join()
        
    print("\nTuning complete!")

    df = study.trials_dataframe()
    df.to_parquet(STUDY_OUTPUT_DIR / f"{STUDY_ARGS['study_name']}.parquet")
    print("\nSaved trials dataframe")

    print("\nBest Hyperparams:")
    best_params = study.best_params
    pp = pprint.PrettyPrinter(indent=4, width=60)
    pp.pprint(best_params)


    # --- Hyperparameter optimization finished


    print("\nTraining Final Model")

    # Setting device to first GPU for final eval
    device = torch.device("cuda:0")   

    # --- DataLoader ---
    final_train_dataset = hyperparam_train_dataset
    final_test_dataset = DATASET_CLASS_TYPE(X_test_tensor, y_test_tensor)
    final_train_loader = DataLoader(final_train_dataset,
                                    batch_size=best_params["batch_size"],
                                    shuffle=True,
                                    num_workers=2
    )
    final_test_loader = DataLoader(final_test_dataset,
                                    batch_size=best_params["batch_size"],
                                    shuffle=False,
                                    num_workers=2
    )

    # Getting input/output sizes
    first_X, first_y = final_train_dataset[0]
    input_size = first_X.shape[0] if first_X.dim() > 0 else 1
    output_size = first_y.shape[0] if first_y.dim() > 0 else 1
    # --- Create final model
    final_model = FeedForwardNN(
        input_features=input_size
        ,hidden_units=tuple(best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"]))
        ,output_features=output_size
        ,dropout_rate=best_params["dropout_rate"]
    ).to(device)
    final_optimizer = optim.Adam(final_model.parameters(),
                                 lr=best_params["learning_rate"],
                                 weight_decay=best_params["weight_decay"]
    )
    final_criterion = nn.BCEWithLogitsLoss()

    # --- Final training loop
    epoch_stats = {
        "train_loss": []
        ,"test_loss": []
        ,"train_acc": []
        ,"test_acc": []
    }

    for epoch in range(best_params["epochs"]):
        final_model.train()
        for batch_X, batch_y in final_train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            final_optimizer.zero_grad()
            outputs = final_model(batch_X)
            # Calculate primary loss (i.e. w/o penalty loss)
            loss = final_criterion(outputs, batch_y)

            # --- Elasic net penalty term ---
            l1_ratio = best_params["l1_ratio"]
            penalty_weight = best_params["penalty_weight"]
            penalty_loss = 0
            for param in final_model.parameters():
                if param.requires_grad:
                    # `param` is a tensor of weights 
                    # (can be a tensor of bias terms as well)
                    penalty_loss += \
                    ( l1_ratio * torch.sum(torch.abs(param)) ) \
                    + ( ((1-l1_ratio) / 2) * torch.sum(torch.square(param)) )

            loss += penalty_weight * penalty_loss

            loss.backward()
            final_optimizer.step() 
        # --- Finished an epoch of training
        
        # --- Eval at every set number of epochs
        if not epoch % 1:
            final_model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # --- Evaluate on Train Set
                train_num_correct_list = []
                train_loss = 0
                for batch_X_train, batch_y_train in final_train_loader:
                    batch_X_train, batch_y_train = batch_X_train.to(device), batch_y_train.to(device)
                    batch_y_train_pred_logits = final_model(batch_X_train).squeeze()
                    batch_avg_loss_train = final_criterion(batch_y_train_pred_logits, batch_y_train)
                    batch_y_train_pred = (torch.sigmoid(batch_y_train_pred_logits) > 0.5).float()
                    batch_train_num_correct = (batch_y_train_pred == batch_y_train).sum().item()
                    train_num_correct_list.append(batch_train_num_correct)
                    train_loss += batch_avg_loss_train.item()
                    
                train_accuracy = sum(train_num_correct_list) / len(final_train_dataset)
                
                # --- Evaluate on Test Set
                test_num_correct_list = []
                test_loss = 0
                for batch_X_test, batch_y_test in final_test_loader:
                    batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                    batch_y_test_pred_logits = final_model(batch_X_test).squeeze()
                    batch_avg_loss_test = final_criterion(batch_y_test_pred_logits, batch_y_test)
                    batch_y_test_pred = (torch.sigmoid(batch_y_test_pred_logits) > 0.5).float()
                    batch_test_num_correct = (batch_y_test_pred == batch_y_test).sum().item()
                    test_num_correct_list.append(batch_test_num_correct)
                    test_loss += batch_avg_loss_test.item()
                    
                test_accuracy = sum(test_num_correct_list) / len(final_test_dataset)
                
                # --- Record stats
                epoch_stats["train_loss"].append(train_loss)
                epoch_stats["train_acc"].append(train_accuracy) 
                epoch_stats["test_loss"].append(test_loss)
                epoch_stats["test_acc"].append(test_accuracy)
        # --- Finished one epoch entirely

    # --- Finished all epochs


    # --- Getting final Test Set info ---
    final_test_loss = epoch_stats["test_loss"][-1]
    final_test_accuracy = epoch_stats["test_acc"][-1]

    print("\nSaving final model epoch stats")
    epoch_stats_df = pd.DataFrame(epoch_stats)
    epoch_stats_df.to_parquet(STUDY_OUTPUT_DIR / "epoch_stats.parquet")
    

    # --- Plotting epoch stats
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax[0].set_title("Loss vs. Epoch")
    ax[0].plot(epoch_stats_df.index, epoch_stats_df["train_loss"], color='blue', label="Train")
    ax[0].plot(epoch_stats_df.index, epoch_stats_df["test_loss"], color='red', label="Test")
    ax[0].legend()

    ax[1].set_title("Accuracy vs. Epoch")
    ax[1].plot(epoch_stats_df.index, epoch_stats_df["train_acc"], color='blue', label="Train")
    ax[1].plot(epoch_stats_df.index, epoch_stats_df["test_acc"], color='red', label="Test")
    ax[1].legend()

    # --- Saving epoch stats plot
    fig.tight_layout(pad=1)
    fig.savefig(STUDY_OUTPUT_DIR / "epoch_stats_plot.png", dpi=100)
    plt.close(fig)


    # --- Printing final test set accuracy
    print(f"\nFinal Test Set Accuracy: {final_test_accuracy}")



if __name__ == "__main__":
    main()