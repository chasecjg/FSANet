<div align="center">

# 🎯 FASNet
## High-Precision Dichotomous Image Segmentation with Frequency and Scale Awareness


<!-- 修正徽章仓库名错误 + 仅保留核心徽章 + 补充TNNLS标识 -->
[![GitHub Stars](https://img.shields.io/github/stars/chasecjg/FSANet?style=for-the-badge&logo=github&color=ff69b4&label=Stars)](https://github.com/chasecjg/FSANet)
[![GitHub License](https://img.shields.io/github/license/chasecjg/FSANet?style=for-the-badge&color=4169e1&label=License)](https://github.com/chasecjg/FSANet/blob/main/LICENSE)
[![TNNLS 2024](https://img.shields.io/badge/TNNLS-2024-007396?style=for-the-badge&logo=IEEE)](https://ieeexplore.ieee.org/journal/189)

</div>

---

## 📋 1. Preface
- This repository provides code for **"High-Precision Dichotomous Image Segmentation with Frequency and Scale Awareness"**, TNNLS 2024
- 📄 [Paper Link](https://ieeexplore.ieee.org/document/10638122)
---

## 📌 2. Overview

### 2.1. Introduction
Dichotomous Image Segmentation (DIS) with rich fine-grained details within a single image is a challenging task. Despite the plausible results achieved by deep learning-based methods, most of them fail to segment generic objects when the boundary is cluttered with the background. In fact, the gradual decrease in feature map resolution during the encoding stage and the misleading texture clue may be the main issues. To handle these issues, we devise a novel frequency- and scale-aware deep neural network (FSANet) for high-precision DIS. The core of our proposed FSANet is twofold. First, a multi-modality fusion (MF) module that integrates the information in spatial and frequency domains is adopted to enhance the representation capability of image features. Second, a collaborative scale fusion module (CSFM), which deviates from the traditional serial structures, is introduced to maintain high resolution during the entire feature encoding stage. On the decoder side, we introduce hierarchical context fusion (HCF) and selective feature fusion (SFF) modules to infer the segmentation results from the output features of the CSFM module. We conduct extensive experiments on several benchmark datasets and compare our proposed method with existing SOTA methods. The experimental results demonstrate that our FSANet achieves superior performance both qualitatively and quantitatively.

---

### 2.2. Framework Overview
<div align="center">
  <img src="https://raw.githubusercontent.com/chasecjg/FSANet/main/Figures/FSANet.jpg" alt="FSANet Architecture" width="85%">
  <br>
  <em>Figure 1: Architecture Overview</em>
</div>


---

### 2.3. Qualitative Results
<div align="center">
  <img src="https://raw.githubusercontent.com/chasecjg/FSANet/main/Figures/Qualitative_comparison.jpg" alt="Qualitative Results" width="85%">
  <br>
  <em>Figure 2: Qualitative Results.</em>
</div>




---
## 3 Quick Start
> **Experimental Setup**: The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with double 3090 GPU of 24 GB Memory.
---
### 3.1. Configuring your environment (Prerequisites):
```bash
# Clone the repository
git clone https://github.com/chasecjg/FSANet.git
cd FSANet

# Create and activate a virtual environment
conda create -n FSANet python=3.12
conda activate FSANet

# Install dependencies
pip install -r requirements.txt
```
---


### 3.2 Downloading necessary data
| Data Type | Target Path | Download Links |
|:---------|:------------|:---------------|
| 📁 Training/Testing Dataset | `./data/` | [Google Drive](https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view?usp=sharing) |
| 📦 Res2Net Pre-trained Weights | `./models/res2net50_v1b_26w_4s-3cf99910.pth` | [Google Drive](https://drive.google.com/file/d/1ITW3_ZBBv2JTviskxO9zfiqlaQ9Nlj-J/view?usp=sharing), [百度网盘](https://pan.baidu.com/s/11KWZfuCU15GC6tUxUxX4Nw?pwd=BCMN) |
| 📦 PVT-v2 Pre-trained Weights | `./FSANet/pvt_v2_b2.pth` | [Google Drive](https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=drive_link), [百度网盘](https://pan.baidu.com/s/1TgxxOGSTuEk4jTtgkZIewA?pwd=1ysw) |
|📦 FSANet Pre-trained Weights|./checkpoints/FSANet/FSANet-100.pth|[Google Drive](https://drive.google.com/file/d/1b0Dfrbxllnl0aChXH8roiDiiirnUpwjq/view?usp=sharing), [百度网盘](https://pan.baidu.com/s/1U0ZYDBFWHJubFs1vln0CNQ?pwd=6666)|

> ⚠️ **Attention**: Ensure the file names and paths are strictly consistent with the above to avoid runtime errors.

---

### 3.3 Training Configuration
⚙️ **Key Operation**: Modify custom path parameters in `train.py`, including:

> - `--train_save`: Path to save training checkpoints and logs
>
> - `--train_path`: Path to the training dataset directory


### 3.4 Testing Configuration
🧪 **Execution Steps**:
> 1. Prepare the pre-trained model and testing dataset
>
> 2. Replace the `--pth_path` parameter in `test.py` with your trained model directory
> 3. Run the test script to generate prediction maps:

```bash
python test.py --pth_path "path/to/your/trained/model"
```


### 3.5 Model Comparison
📊 **Visualization Data**: We provide visualization images of our model that you can download from:
> - [Google Drive](https://drive.google.com/file/d/1YulLz9L9dBKUdOFn-3zmp2Xg88UlAzr2/view?usp=drive_link)
> - [Baidu Netdisk](https://pan.baidu.com/s/1Q5fEdBRH_xVkagVribM9Hg?pwd=bix2)

---

## 4 Evaluating your trained model:
> **Evaluation Tool**: One-key evaluation is written in Python code (revised from [link](https://github.com/lartpang/PySODMetrics))






---
## 5. Citation
````bibtex
@ARTICLE{JinguangchengFSANet,
  author={Jiang, Qiuping and Cheng, Jinguang and Wu, Zongwei and Cong, Runmin and Timofte, Radu},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={High-Precision Dichotomous Image Segmentation With Frequency and Scale Awareness}, 
  year={2025},
  volume={36},
  number={5},
  pages={8619-8631},
  doi={10.1109/TNNLS.2024.3426529}}
````

---
## 📫 Welcome to star, fork and collaborate!

**[⬆ back to top](#1-preface)**
