import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

from .config import GLOBAL_RANDOM_STATE


class TabularDataset(Dataset):
    """
    Custom dataset object that is for tabular data only.
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

def load_data(path: str, 
              scale_features: bool, 
              scale_targets: bool,
              train_ratio: float
    ) -> Tuple[torch.Tensor, 
               torch.Tensor,
               torch.Tensor,
               torch.Tensor
               ]:
    """
    Reads in data from external source (disk, database, etc.) 
    and returns features and labels as Pytorch tensors.

    Can modify this depending on dataset your using and its source.
    """
    df = pd.read_parquet(path)
    feature_cols = df.columns[0:-1]
    label_cols = df.columns[-1]

    X_train, X_test, y_train, y_test = \
    train_test_split(
        df[feature_cols],
        df[label_cols],
        train_size=train_ratio,
        shuffle=True,
        random_state=GLOBAL_RANDOM_STATE
    )

    # --- Scaling Features
    if scale_features:
        reshaped_X = False
        if len(X_train.shape) == 1:
            X_train = X_train.values.reshape(-1, 1)
            X_test = X_test.values.reshape(-1, 1)
            reshaped_X = True
        
        labels_scaler = StandardScaler()
        X_train = labels_scaler.fit_transform(X_train)
        X_test = labels_scaler.transform(X_test)
        
        # if we reshape X into an Nx1 ndarray, now we flatten
        if reshaped_X:
            X_train = X_train.ravel()
            X_test = X_test.ravel()
    else:
        X_train = X_train.values
        X_test = X_test.values

    # --- Scaling Labels
    if scale_targets:
        reshaped_y = False
        if len(y_train.shape) == 1:
            y_train = y_train.values.reshape(-1, 1)
            y_test = y_test.values.reshape(-1, 1)
            reshaped_y = True
        
        labels_scaler = StandardScaler()
        y_train = labels_scaler.fit_transform(y_train)
        y_test = labels_scaler.transform(y_test)
        
        # if we reshape y into an Nx1 ndarray, now we flatten
        if reshaped_y:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
    else:
        y_train = y_train.values
        y_test = y_test.values
    

    # --- Converting to tensors
    X_train_tensor = torch.from_numpy(X_train).to(torch.float32)
    X_test_tensor = torch.from_numpy(X_test).to(torch.float32)

    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)


    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor