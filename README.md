# nn-pipelines

## Setup environment
Consists of two main parts: 
- creating the conda environment and 
- then installing pip dependencies

This is done so as to install pytorch properly from the pytorch official website using pip.

**1.** Create the conda environment with non-pip dependencies
```bash
conda create -f environment.yml
``` 

**2.** Install the pip dependencies 
```bash
pip install -r pip_requirements.txt
```

*Important:* Ensure that the `pip_requirements.txt` file has the `--extra-index-url https://download.pytorch.org/whl/cu128` above the `torch` and `torchvision` pip dependencies.



## Export conda environment to yml file
Follow this if you need to add dependencies to the environment.

```bash
conda env export | head -n -1 > environment.yml
```
The `head -n -1` command removes the last line of text which is added by conda by default. It is a prefix label (e.g. `prefix: /path/to/miniconda3/envs/nn-pipelines`)

## Checking CUDA version on HPC
On an HPC GPU node, if the `nvcc --version` command doesn't work, you might need to load the `cuda-toolkit/<version_num>` application into your node session with the following command
```bash
module load cuda-toolkit/<version_num>
``` 
Then you will be able to run the `nvcc --version` command that checks the version of the latest nvidia cuda compiler.

## Checking available GPU nodes and the GPU models on them
```bash
$ sinfo -o "%N %G"

NODELIST GRES
compute-11-[24,26-30],compute-12-[00-26,28-47],compute-14-[00-22,24-47],compute-15-[00-07],compute-17-[00-47],compute-33-[02-11],compute-34-[00-02,04-31],compute-41-[01-25] (null)
gpu-42-00 gpu:H100:8(S:0-1)
gpu-15-[23-24,27-28,31-32] gpu:V100-SXM2-32GB:4(S:0-1)
gpu-33-28 gpu:V100-PCIE-32GB:2(S:0-1),shard:V100-PCIE-32GB:40(S:0-1)
```

Use this name e.g. `gpu:V100-SXM2-32GB:3` when submitting a job and requesting resources (in this case 3 V100-SXM2-32GB GPUs). For example, `salloc --time=1:00:00 --gres=gpu:V100-SXM2-32GB:3 --partition=gpu`
