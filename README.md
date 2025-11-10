# UnReL: Unlearning via Relearning

This repository contains the reproducibility code for **UnReL**, an experimental framework exploring efficient data unlearning in large language models (LLMs).  
The project focuses on estimating how likely each data instance is to be requested for removal and preparing models for targeted retraining with minimal cost.

---

## 🚀 Overview

UnReL works in two broad steps:
1. **Risk Scoring:** Each file or dataset item is analyzed to estimate its likelihood of future unlearning requests.  
2. **Shard Assignment:** Data are ordered and distributed into shards for training and evaluation under different strategies.

The system uses a Sentence-Transformer model to assess semantic similarity to sensitive categories (e.g., personal, medical, financial) and combines that with temporal indicators.

---

## Installation

To set up the environment for the project, create a conda environment using the following command:

```bash
$ conda create --name torch-env pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
$ conda activate torch-env
```

Then, install the following libraries:

```bash
pip install datasets accelerate evaluate matplotlib hydra-core omegaconf peft rouge_score tqdm einops packaging bitsandbytes scipy ninja
```
Also, you may need to install additional updated libraries if required. 

## Traditional Retraining from Scratch (Reference baseline)

To perform traditional retraining from scratch, run the following command:

```bash
python finetune.py --config-path /home/user_name/project_name/config --config-name finetune.yaml
```
Do necessary modification in finetune.yaml file based on your hardware and GPU capacity.

## Running the AutoScoring module for sharding data based on the likelihood of forgetting

Run the scoring module provided dataset directory inside the main function of the code:

```bash
python autoscoring.py
```

## Unlearning Ready Training

To train the model under UnReL, SISA, and APA, use the following command:

For UnReL 
```bash
python UnReL_training.py --config-path /home/user_name/project_name/config --config-name UnReL_training.yaml
```
For SISA
```bash
python SISA_training.py --config-path /home/user_name/project_name/config --config-name Train_dp_SGD.yaml
```
Do necessary modification in Train_dp_MLM.yaml or Train_dp_SGD.yaml based on your hardware and GPU capacity. 

## DP2Unlearning Fine-Tuning

For DP2Unlearning fine-tuning, run:

```bash
python FT_BaseModel.py --config-path /home/user_name/project_name/config --config-name FT_BaseModel.yaml
```
Do necessary modification to FT_BaseModel.yaml based on forgetting percentage (1%:retain99, 5%:retain95, or 10%:retain90)

## Approximate Unlearning Baselines Fine-Tuning

To perform approximate unlearning fine-tuning, execute the following:

```bash
python forget.py --config-path /home/user_name/project_name/config --config-name forget.yaml
```

## Evaluation

To evaluate the models, use this command:

```bash
python evaluate_util.py --config-path /home/user_name/project_name/config --config-name eval_everything.yaml
```
You need to provide the specific model path that you wish to evaluate. 

## Aggregation

To aggregate the evaluation statistics, use:

```bash
python aggregate_eval_stat.py --config-path /home/user_name/project_name/config --config-name aggregate_eval_stat.yaml
```

Ensure you have the paths to your results:

```bash
retain_result=${path_to_traditional_retraining_from_scratch}
ckpt_result=${path_to_your_unlearned_method}
```

## Beyond KS Test

To run the Beyond KS Test, execute:

```bash
python Beyond_KS_test.py --config-path /home/user_name/project_name/config --config-name aggregate_eval_stat.yaml
```
---------------------------------------------------------------------------------------------------------------------

<small>The baseline methods are implemented from </small> [[1](https://locuslab.github.io/tofu/)]


## ⚙️ Running the AutoScoring Tool

Run the scoring module on your dataset directory:

```bash
python autoscoring.py
