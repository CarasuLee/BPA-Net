# BPA-Net: Boosting Image Captioning with Prototype-based Alignment

This repository contains the implementation of **BPA-Net**, a novel image captioning framework designed to improve cross-modal alignment using learnable semantic prototypes.

## 🚀 Framework Overview

![Framework Architecture](introduction.png)

BPA-Net introduces:
- **Learnable Semantic Prototypes**: For capturing high-level semantic concepts.
- **VPE (Visual Prototype Encoding) Module**: Aggregates patch features into semantic-aware visual representations.
- **CMPA (Cross-Modal Prototype Alignment) Module**: Ensures consistency between visual and textual prototypes.
- **TAP (Textual Alignment Prototype) Module**: Aligns textual features with semantic prototypes.

## 🛠 Installation

### 1. Requirements
- Python 3.10+
- PyTorch 1.12+ 
- CUDA 11.3+ 
- h5py
- spacy
- tqdm
- numpy

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/LiTianci2024/BPANet.git
cd BPANet

# Install basic dependencies
pip install -r requirements.txt

# Download Spacy model
python -m spacy download en_core_web_sm
```

## 🖼 CLIP Configuration

BPA-Net supports both **OpenAI CLIP** and **HuggingFace Transformers CLIP** as backbones.

### Option A: OpenAI CLIP (Official)
Install the CLIP library from the official repository:
```bash
pip install git+https://github.com/openai/CLIP.git
```
In `train_transformer.py`, you can specify the model name (e.g., `ViT-L/14`, `ViT-B/32`):
```bash
python train_transformer.py --clip_model_path ViT-L/14 ...
```
The model will be automatically downloaded to `~/.cache/clip`.

### Option B: HuggingFace Transformers
If you prefer using pre-trained models from HuggingFace:
```bash
pip install transformers
```
Specify the HF repository name or local path:
```bash
python train_transformer.py --clip_model_path openai/clip-vit-large-patch14 ...
```

## 📊 Training & Evaluation

### Data Preparation
Please prepare the COCO/Flickr30k datasets and extract features if needed. Ensure paths are correctly set in the command line arguments.

### Training with Cross-Entropy
```bash
python train_transformer.py \
    --img_root_path /path/to/images \
    --annotation_folder /path/to/annotations \
    --features_path /path/to/features \
    --exp_name COCO_BPA
```

### Training with Self-Critical (RL)
The script will automatically switch to the RL stage after patience is reached or use `--resume_best` with specific RL flags.

## 📜 Citation
If you find this work useful, please cite our paper.
