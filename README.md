# Fast Video Temporal Grounding / Fast Video Moment Retrieval
Code for paper: **Learning Commonsense-aware Moment-Text Alignment for Fast Video Temporal Grounding**.

## Introduction

### Abstract:
Grounding temporal video segments described in natural language queries effectively and efficiently is a crucial capability needed in vision-and-language fields. In this paper, we deal with the fast video temporal grounding (FVTG) task, aiming at localizing the target segment with high speed and favorable accuracy. Most existing approaches adopt elaborately designed cross-modal interaction modules to improve the grounding performance, which suffer from the test-time bottleneck. Although several common space-based methods enjoy the high-speed merit during inference, they can hardly capture the comprehensive and explicit relations between visual and textual modalities. In this paper, to tackle the dilemma of speed-accuracy tradeoff, we propose a commonsense-aware cross-modal alignment (CCA) framework, which incorporates commonsense-guided visual and text representations into a complementary common space for fast video temporal grounding. Specifically, the commonsense concepts are explored and exploited by extracting the structural semantic information from a language corpus. Then, a commonsense-aware interaction module is designed to obtain bridged visual and text features by utilizing the learned commonsense concepts. Finally, to maintain the original semantic information of textual queries, a cross-modal complementary common space is optimized to obtain matching scores for performing FVTG. Extensive results on two challenging benchmarks show that our CCA method performs favorably against state-of-the-arts while running at high speed.

## Download data
**Download the dataset files and pre-trained model files from [here](https://drive.google.com/drive/folders/1vpJWo7ZtVgrVyKQrtd8kfcY7GZeVYD0V?usp=sharing).**

## Training and Inference
We provide scripts for simplifying training and inference. Please modify corresponding file and run it.

**Training on TACoS:**  [script/train_acnet.sh](script/train_acnet.sh).

**Training on ActivityNet Captions:** [script/train.sh](script/train.sh).

**Evaluating on TACoS:** [script/eval.sh](script/eval.sh)

**Evaluating on ActivtyNet Captions:** [script/eval_acnet.sh](script/eval_acnet.sh)

## Citation
Please **Star** this repo and **cite** the following paper if you feel our CCA useful to your research:

```
@misc{2204.01450,
Author = {Ziyue Wu and Junyu Gao and Shucheng Huang and Changsheng Xu},
Title = {Learning Commonsense-aware Moment-Text Alignment for Fast Video Temporal Grounding},
Year = {2022},
Eprint = {arXiv:2204.01450},
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
