# Fast Video Temporal Grounding / Fast Video Moment Retrieval
Code for paper: **Learning Commonsense-aware Moment-Text Alignment for Fast Video Temporal Grounding**.

**Paper is available [here](https://arxiv.org/pdf/2204.01450.pdf)**.

*Ziyue Wu, Junyu Gao, Shucheng Huang, Changsheng Xu*

## Introduction

### Abstract:
Grounding temporal video segments described in natural language queries effectively and efficiently is a crucial capability needed in vision-and-language fields. In this paper, we deal with the fast video temporal grounding (FVTG) task, aiming at localizing the target segment with high speed and favorable accuracy. Most existing approaches adopt elaborately designed cross-modal interaction modules to improve the grounding performance, which suffer from the test-time bottleneck. Although several common space-based methods enjoy the high-speed merit during inference, they can hardly capture the comprehensive and explicit relations between visual and textual modalities. In this paper, to tackle the dilemma of speed-accuracy tradeoff, we propose a commonsense-aware cross-modal alignment (CCA) framework, which incorporates commonsense-guided visual and text representations into a complementary common space for fast video temporal grounding. Specifically, the commonsense concepts are explored and exploited by extracting the structural semantic information from a language corpus. Then, a commonsense-aware interaction module is designed to obtain bridged visual and text features by utilizing the learned commonsense concepts. Finally, to maintain the original semantic information of textual queries, a cross-modal complementary common space is optimized to obtain matching scores for performing FVTG. Extensive results on two challenging benchmarks show that our CCA method performs favorably against state-of-the-arts while running at high speed.

### Overview of the proposed method:
![framework_00](https://user-images.githubusercontent.com/102899678/161745291-fb0654f5-5e0b-46c0-b610-015b095c040f.png)


## Download data
**Download the dataset files and pre-trained model files from [here](https://drive.google.com/drive/folders/1vpJWo7ZtVgrVyKQrtd8kfcY7GZeVYD0V?usp=sharing).**

## Training and Inference
We provide scripts for simplifying training and inference. Please modify corresponding file and run it.

**Training on TACoS:** [script/train.sh](script/train.sh).

**Training on ActivityNet Captions:** [script/train_acnet.sh](script/train_acnet.sh).

**Evaluating on TACoS:** [script/eval.sh](script/eval.sh)

**Evaluating on ActivtyNet Captions:** [script/eval_acnet.sh](script/eval_acnet.sh)

## Main Results

### 1.Speed-Accuracy on TACoS

|Methods|TE|CML|ALL|ACC|sumACC|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FVMR|3.51|0.14|3.65|29.12|70.60|
|**CCA(ours)|2.33|0.29|2.62|32.83|78.13|**

### 2.Speed-Accuracy on ActivityNet Captions

|Methods|TE|CML|ALL|ACC|sumACC|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FVMR|3.14|0.09|3.23|45.00|106.60|
|**CCA(ours)|2.80|0.30|3.10|46.19|106.77|**

### 3.TACoS with C3D features

|R@1,IoU=0.1|R@1,IoU=0.3|R@1,IoU=0.5|R@1,IoU=0.7|R@5,IoU=0.1|R@5,IoU=0.3|R@5,IoU=0.5|R@5,IoU=0.7|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|56.00|45.30|32.83|18.07|76.60|64.38|52.68|33.10|

### 4.ActivityNet Captions with C3D features

|R@1,IoU=0.3|R@1,IoU=0.5|R@1,IoU=0.7|R@5,IoU=0.3|R@5,IoU=0.5|R@5,IoU=0.7|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|60.58|46.19|28.87|86.02|77.86|60.28|


## Citation
Please **Star** this repo and **cite** the following paper if you feel our CCA useful to your research:

```
@article{wu2022commonsense,
  title={Learning Commonsense-aware Moment-Text Alignment for Fast Video Temporal Grounding},
  author={Wu, Ziyue and Gao, Junyu and Huang, Shucheng and Xu, Changsheng},
  journal={arXiv preprint arXiv:2204.01450},
  year={2022}
}
```

```
@inproceedings{gao2021fast,
  title={Fast video moment retrieval},
  author={Gao, Junyu and Xu, Changsheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1523--1532},
  year={2021}
}
```
