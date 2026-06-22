# EvoSNR-Prom
EvoSNR-Prom is a lexicon-enhanced single-nucleotide resolu-tion predictor with label-aware transfer learning based on EVO pre-trained model for DNA promoter prediction

# Dependency
* Python3
* biopython==1.85
* evo-model==0.4
* fasttext==0.9.3
* flash-attn>=2.7.0
* numpy==1.26.0
* peft==0.15.2
* torch==2.2.2
* torchaudio==2.2.2
* torchvision==0.17.2
* transformers==4.52.4
* triton==2.2.0

# Content
Baseline:This directory contains the baseline models and their corresponding training data used for comparative experiments.
data: data for model training ( data of *Sinorhizobium meliloti 1021* ,  *Agrobacterium tumefaciens strain C58* ,*Escherichia coli str K-12 substr. MG1655*,*Klebsiella aerogenes KCTC 2190* )

# Usage
## 1.Environment setup
We recommend you to build a python virtual environment with Anaconda. We applied training on single NVIDIA GeForce RTX 4090  with 24 GB graphic memory, and the batch size corresponds to it. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

1.1 Create and activate a new virtual environment
```python
conda create -n evosnr python=3.11
conda activate evosnr
```
1.2 Install the package and other requirements
```
pip install -r requirements.txt
```

1.3 download datasets of EvoSNR-Prom
>  Note: The dataset includes large FastText model weights, so it is hosted on Hugging Face Hub instead of being stored directly in this repository.

datasets link: https://huggingface.co/datasets/lightyisu/EvoSNR-Prom
Download the data folder and overwrite the existing data directory in repository .

1.4  Modify train_config.py 

To customize the training setup according to your specific requirements, you need to modify the parameters in the ```train_config.py``` file. 
The key configurations you should update: Dataset Path,Lexicons Path,  FastText Pre-trained Model Path

2.train EvoSNR-Prom
```python
python train_EvoSNR-Prom.py
```
