# Fast Video Temporal Grounding / Fast Video Moment Retrieval
Code for paper: **Learning Commonsense-aware Moment-Text Alignment for Fast Video Temporal Grounding**.

## Introduction

### Abstract:
Grounding temporal video segments described in natural language queries effectively and efficiently is a crucial capability needed in vision-and-language fields. In this paper, we deal with the fast video temporal grounding (FVTG) task, aiming at localizing the target segment with high speed and favorable accuracy. Most existing approaches adopt elaborately designed cross-modal interaction modules to improve the grounding performance, which suffer from the test-time bottleneck. Although several common space-based methods enjoy the high-speed merit during inference, they can hardly capture the comprehensive and explicit relations between visual and textual modalities. In this paper, to tackle the dilemma of speed-accuracy tradeoff, we propose a commonsense-aware cross-modal alignment (CCA) framework, which incorporates commonsense-guided visual and text representations into a complementary common space for fast video temporal grounding. Specifically, the commonsense concepts are explored and exploited by extracting the structural semantic information from a language corpus. Then, a commonsense-aware interaction module is designed to obtain bridged visual and text features by utilizing the learned commonsense concepts. Finally, to maintain the original semantic information of textual queries, a cross-modal complementary common space is optimized to obtain matching scores for performing FVTG. Extensive results on two challenging benchmarks show that our CCA method performs favorably against state-of-the-arts while running at high speed.

## Download data
**Download the dataset files and pre-trained model files from [here](https://drive.google.com/drive/folders/1vpJWo7ZtVgrVyKQrtd8kfcY7GZeVYD0V?usp=sharing).**

## Start
We provide scripts for simplifying training and inference. Please refer to , script/eval.sh and script/eval_acnet.sh.

For training model, please modify ''script/train.sh'' or ''script/train_acnet.sh'' and run:

''python train.sh''

''python train_acnet.sh''
