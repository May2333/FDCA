# Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval

This repo is the official implementation of ICLR 2025 paper "Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval".
<br>
By Yue Wu, Zhaobo Qi, Yiling Wu, Junshu Sun, Yaowei Wang, Shuhui Wang

# Introduction

<p style="text-align:justify; text-justify:inter-ideograph;">
With the explosive growth of video data, finding videos that meet detailed requirements in large datasets has become a challenge. To address this, the composed video retrieval task has been introduced, enabling users to retrieve videos using complex queries that involve both visual and textual information. However, the inherent heterogeneity between modalities poses significant challenges. Textual data is highly abstract, while video content contains substantial redundancy. This modality gap in information representation makes existing methods struggle with the fine-grained fusion and alignment required for fine-grained composed retrieval. To overcome these challenges, we introduce FineCVR-1M, a fine-grained composed video retrieval dataset containing 1,010,071 video-text triplets with detailed textual descriptions. This dataset is constructed through an automated process that identifies key concept changes between video pairs to generate textual descriptions for both static and action concepts. For fine-grained retrieval methods, the key challenge lies in understanding the detailed requirements. Text descriptions serve as clear expressions of intent, allowing models to distinguish fine-grained needs through textual feature disentanglement. Therefore, we propose a textual Feature Disentanglement and Cross-modal Alignment framework FDCA that disentangles features at both the sentence and token levels. At the sequence level, we separate text features into retained and injected features. At the token level, an Auxiliary Token Disentangling mechanism is proposed to disentangle texts into retained, injected, and excluded tokens. The disentanglement at both levels extracts fine-grained features, which are aligned and fused with reference video to extract global representations for video retrieval. Experiments on FineCVR-1M dataset demonstrate the superior performance of FDCA.
</p>

# Installation

## Environment Setup
```bash
# Create and activate conda environment
conda create -n combiner python=3.8 -y
conda activate combiner

# Install required packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# Data Preparation

## Original Video Frames
We will upload original video frames as soon as possible.

## Directory Structure Setup

(1) Set dataset path environment variable:
```bash
export FINECVR_DATASET_ROOT=/your/actual/dataset/path
```
(2) Download annotation files from [Google Drive](https://drive.google.com/drive/folders/1SneQu9pUhvWmehGxn_Y8YB0JGaa-XfAv?usp=drive_link) and place them in:
```
$FINECVR_DATASET_ROOT/FineCVR/annotations/
```
(3) Download pre-extracted CLIP features from [Google Drive](https://drive.google.com/drive/folders/1m6zM0udCj8LThWsiMAtsQBFEWAmMucTI?usp=drive_link):
```
# Extract to target directory (ensure tar file is downloaded first)
tar -xf CLIP_RN50x4_high_8_640.tar -C $FINECVR_DATASET_ROOT/FineCVR/embeddings/
```
(4) Final directory structure should be:

```
$FINECVR_DATASET_ROOT/
└── FineCVR/
    ├── annotations/
    │   ├── train.txt
    │   ├── test.txt
    │   ├── train_remaped.txt
    │   ├── test_remaped.txt
    │   ├── id2vdoname_train.json
    │   ├── id2vdoname_test.json
    │   ├── vdoname2id_train.json
    │   └── vdoname2id_test.json
    └── embeddings/
        └── CLIP_RN50x4_high_8_640/
```

# Training
Train FDCA:
```
python combiner_train.py \
    --dataset FineCVR \
    --data_pth $FINECVR_DATASET_ROOT/FineCVR \
    --save-best \
    --save-training
```

# Validation
Validate with pretrained model (ensure model file exists):
```
python validate.py \
    --data_pth $FINECVR_DATASET_ROOT/FineCVR \
    --combiner-path saved_models/combiner_arithmetic.pt  # Update to actual model path
```


# Cite

If you use our dataset or method in your research, please cite our paper:
```
 @article{yue25finecvr,
        title = {Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval},
        author = {Yue Wu, Zhaobo Qi, Yiling Wu, Junshu Sun, Yaowei Wang, Shuhui Wang},
        journal = {ICLR},
        year = {2025}
    }
```

# Acknowledgement
This repository is built based on [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir) repository.
