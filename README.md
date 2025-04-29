# drug-detector

# I. Project Setup Guide

## Prerequisites

Ensure you have installed **Python 3.10.16** and **Updated the latest version of GPU driver** before proceeding.

### Create Virtual Environment
```sh
python -m venv venv
```

### Upgrade Pip
```sh
pip install --upgrade pip
```

### Activate the virtual environment
```sh
source venv/bin/activate
```

## Install CUDA
- Check you're GPU CUDA version
```
nvidia-smi
```
- Then go to https://pytorch.org/get-started/locally/ and get the correct install command.
- For Example:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Install YOLO8
- Install Yolo version 8
```
pip install ultralytics
```
- Verify GPU installation by script:
```
python workspace/src/verify_yolo.py
```

## Training Model
- Use python script
```
python workspace/src/train_model.py --help for more details
```