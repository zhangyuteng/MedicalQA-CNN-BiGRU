# MedicalQA-CNN-BiGRU

This repo contains the implementation of "Chinese Medical Question Answer Selection via Hybrid Models based on CNN and GRU" in PyTorch.
It implements five models, namely, stack-CNN, multi-CNN, multi-stack-CNN, BiGRU, BiGRU-CNN.


## Usage for python code

### 0. Requirement

python 3.6  
numpy==1.14.5  
pandas==0.22.0  
tensorboard==1.8.0  
tensorboardX==1.4  
tensorflow==1.8.0  
torch==0.4.1  
torchtext==0.3.0  
tqdm==4.7.2

```bash
pip install -r requirements.txt
```

### 1. Data preparation

The dataset is [cMedQA](https://github.com/zhangsheng93/cMedQA)

```bash
git submodule update --init --recursive
chmod +x preproc.sh
bash preproc.sh
```

### 2. Start the training process

See help.

```bash
python train.py --help
```

Basic use
```bash
python train.py --arch bigru_cnn --device 0 --batch-size 64 --epoch 5 --hidden-size 200
```