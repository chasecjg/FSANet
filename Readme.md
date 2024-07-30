# High-Precision Dichotomous Image Segmentation with Frequency and Scale Awareness (TNNLs-2024)

> **Authors:** 
> Qiuping Jiang,
> Jinguang Cheng<sup>*</sup>,
> Zongwei Wu,
> Qiuping Jiang,
> Radu Timofte.

## 1. Preface

- This repository provides code for "_**High-Precision Dichotomous Image Segmentation with Frequency and Scale Awareness**_" TNNLs-2024.

## 2. Overview

### 2.1. Introduction
Dichotomous Image Segmentation (DIS) with rich fine-grained details within a single image is a challenging task. Despite the plausible results achieved by deep learning-based methods, most of them fail to segment generic objects when the boundary is cluttered with the background. In fact, the gradual decrease in feature map resolution during the encoding stage and the misleading texture clue may be the main issues. To handle these issues, we devise a novel frequency- and scale-aware deep neural network (FSANet) for high-precision DIS. The core of our proposed FSANet is twofold. First, a multi-modality fusion (MF) module that integrates the information in spatial and frequency domains is adopted to enhance the representation capability of image features. Second, a collaborative scale fusion module (CSFM) which deviates from the traditional serial structures is introduced to maintain high resolution during the entire feature encoding stage. In the decoder side, we introduce hierarchical context fusion (HCF) and selective feature fusion (SFF) modules to infer the segmentation results from the output features of the CSFM module. We conduct extensive experiments on several benchmark datasets and compare our proposed method with existing SOTA methods. The experimental results demonstrate that our FSANet achieves superior performance both qualitatively and quantitatively. 


### 2.2. Framework Overview

<p>
  <img src="https://raw.githubusercontent.com/chasecjg/FSANet/main/Figures/FSANet.jpg" alt="FSANet Architecture">
  <br>
  <em>Figure 1: Architecture Overview</em>
</p>

### 2.3. Qualitative Results

<p>
    <img src="https://raw.githubusercontent.com/chasecjg/FSANet/main/Figures/Qualitative_comparison.jpg" alt="Qualitative Results">
    <br>
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
double 3090 GPU of 24 GB Memory.

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n FSANet python=3.8`.
    
    + Installing necessary packages: `pip install -r requirements.txt`. (PS: You can also select some of these packages to install that you need.)

1. Downloading necessary data:

    + downloading training/testing dataset and move it into `data`, 
    which can be found in this [(Google Drive)](https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view?usp=sharing).
    


    + downloading Res2Net weights and move it into `./models/res2net50_v1b_26w_4s-3cf99910.pth`[(Google Drive)](https://drive.google.com/file/d/1ITW3_ZBBv2JTviskxO9zfiqlaQ9Nlj-J/view?usp=sharing) or [(BaiduNetdisk)](https://pan.baidu.com/s/11KWZfuCU15GC6tUxUxX4Nw?pwd=BCMN) 
    + downloading pvt-v2 weights and move it into `./FSANet/pvt_v2_b2.pth`[(Google Drive)](https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=drive_link) or [(BaiduNetdisk)](https://pan.baidu.com/s/1TgxxOGSTuEk4jTtgkZIewA?pwd=1ysw) 

1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `train.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

1. Comparing with your model:
    + We provide visualizations images that you can download form [(Google Drive)](https://drive.google.com/file/d/1YulLz9L9dBKUdOFn-3zmp2Xg88UlAzr2/view?usp=drive_link) or [(Baidudisk)](https://pan.baidu.com/s/1Q5fEdBRH_xVkagVribM9Hg?pwd=bix2)
 
### 3.2 Evaluating your trained model:

One-key evaluation is written in python code (revised from [link](https://github.com/lartpang/PySODMetrics))


## 4. Citation


**[⬆ back to top](#1-preface)**
