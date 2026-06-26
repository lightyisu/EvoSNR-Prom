# EvoSNR-Prom

EvoSNR-Prom is a lexicon-enhanced single-nucleotide resolu-tion predictor with label-aware transfer learning based on EVO pre-trained model for DNA promoter prediction

# Dependency

- Python 3.11
- biopython 1.87
- evo-model 0.5
- fasttext 0.9.3
- flash_attn 2.7.4.post1
- Jinja2 3.1.6
- numpy 2.4.6
- peft 0.19.1
- torch 2.6.0
- torchaudio 2.6.0
- torchvision 0.21.0
- tqdm 4.68.3
- transformers 5.12.1
- triton 3.2.0

# Content

Baseline:This directory contains the baseline models and their corresponding training data used for comparative experiments.
data: data for model training ( data of _Sinorhizobium meliloti 1021_ , _Agrobacterium tumefaciens strain C58_ ,_Escherichia coli str K-12 substr. MG1655_,_Klebsiella aerogenes KCTC 2190_ )

# Usage

## 1.Environment setup

We recommend you to build a python virtual environment with Anaconda. We applied training on single NVIDIA GeForce RTX 4090 with 24 GB graphic memory, and the batch size corresponds to it. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

1.1 Create and activate a new virtual environment

```python
conda create -n evosnr python=3.11
conda activate evosnr
```

1.2 Install the recommended PyTorch version

```python
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

1.2 Prerequisites for installing the Evo package
(see more in https://github.com/evo-design/evo)

```
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge flash-attn=2.7.4
```

You can install flash-attn directly via a .whl file.
Pre-built .whl files are available at https://github.com/Dao-AILab/flash-attention/releases

1.3 Install Evo using pip

```
pip install evo-model
```

1.4 Install the remaining dependency packages

1.5 download datasets of EvoSNR-Prom

> Note: The dataset includes large FastText model weights, so it is hosted on Hugging Face Hub instead of being stored directly in this repository.

datasets link: https://huggingface.co/datasets/lightyisu/EvoSNR-Prom
Download the data folder and overwrite the existing data directory in repository .

1.6 Modify train_config.py

To customize the training setup according to your specific requirements, you need to modify the parameters in the `train_config.py` file.
The key configurations you should update: Dataset Path,Lexicons Path, FastText Pre-trained Model Path

## 2.train EvoSNR-Prom

```python
python train.py
```
