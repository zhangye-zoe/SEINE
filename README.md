# SEINE: Segmentation and Explanation of Interactive Nuclei Embeddings

SEINE is a framework for nuclei segmentation and reasoning in histopathological images. It supports training, evaluation, and interpretation of deep learning models for instance-level nuclei analysis.

## 📁 Dataset Preparation

Please refer to the detailed instructions in [docs/data_prepare.md](docs/data_prepare.md) for dataset organization and formatting.

For preprocessing procedures such as patch extraction, data splitting, and structural encoding, please check [docs/data.ipynb](docs/data.ipynb), which provides step-by-step processing code and explanations.

---

## 🔧 Installation

We recommend using a conda virtual environment.

```bash
# 1. Create environment
conda create -n seine python=3.7 -y
conda activate seine

# 2. Install PyTorch (ensure CUDA 11.1 support)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV-full (Linux recommended)
pip install mmcv-full==1.3.13

# 4. Install required packages
pip install -r requirements.txt

# 5. Clone and install this repo
git clone https://github.com/zhangye-zoe/SEINE.git
cd SEINE
pip install -e .
```
✅ The code has been tested and runs successfully on both NVIDIA RTX 3090 and A100 GPUs.

## 🚀 Usage

### 🔧 Training

To start training:

```bash
# Replace [config_path] with your actual config file
CUDA_VISIBLE_DEVICES=0 python tools/train.py [config_path]
```

### 📊 Evaluation

To evaluate a trained model:

```bash
# Replace [config_path] and [checkpoint] with your config file and checkpoint path
CUDA_VISIBLE_DEVICES=0 python tools/test.py [config_path] [checkpoint]
```

## 📂 Project Structure 

```
SEINE/
├── configs/            # Configuration files
├── data/               # Dataset loading & interface
├── docs/               # Documentation and data preparation guides
├── tiseg/              # Core model architectures and components
├── tools/              # Training, testing, and utilities
├── requirements.txt    # Python dependency list
└── README.md           # Project overview

```

## 🙏 Acknowledgements

This project is inspired by the design pattern of  
[Tissue-Image-Segmentation](https://github.com/sennnnn/Tissue-Image-Segmentation). We thank their contribution to open-source medical image analysis.

## 📬 Contact

For questions, please feel free to contact the author via GitHub Issues or email.
