# Overview

This is the github repository for "Approximate Clustering for Extracting Task Relationships in Multi-Instruction Tuning".

# Requirements
You can either set up a python environment by adding the following key packages:
- Python = 3.8
- pytorch = 2.0.0
- transformers = 4.28.1
- datasets = 2.11.0

and other related packages based on the information,

 or run 

```bash
pip install -r requirements.txt
```


# Usage

The code in this repository provides several ways to fine-tune LM provided in hugging face to estimate the affinity score matrix of different tasks which can be used to help doing clustering of the tasks that have positive effect with each other. More specifically, we can fine-tune LM with 

- Different instructions/prompts (none, single, multiple)
- Different Tasks (single or multiple)


# How to run:

### For the instruction tuning, we provide a script `train_multi_instruction.py`

The follow example will show you how to run an expriment.

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_instruction.py \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --model_name_or_path t5-base \
    --max_source_length 512 \
    --max_target_length 128 \
    --pad_to_max_length True \
    --generation_max_length 128 \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --lr_scheduler_type constant \
    --max_steps 5000 \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --metric_for_best_model eval_exact_match\
    --greater_is_better True \
    --task_name rte \
    --template_idx 0 1 2 3 4 5\
    --output_dir saved/ \
    --load_best_model_at_end \
    --disable_tqdm True 
```

This bash will fine-tune the `t5-base` model with `rte` dataset that adding the `0 1 2 3 4 5` templates of instructions.

The key configuration here are:

- `model_name_or_path` : representing the pretrained model name loaded from hugging face.
- `task_name` : representing the dataset(task) we use to fine-tuning the LM.
- `template_idx` : representing the prompt idx that added to the fine-tuning process.
- `output_dir` : representing the saved path for result.
- `max_steps`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps` are some useful configuration for fintuning, please adjust them when you need to.

### For multitask tuning, we provide a script `train_multitask.py`

The follow example will show you how to run an expriment.

```bash
CUDA_VISIBLE_DEVICES=0 python train_multitask.py \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --model_name_or_path t5-small \
    --max_source_length 512 \
    --max_target_length 128 \
    --pad_to_max_length True \
    --generation_max_length 128 \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --lr_scheduler_type constant \
    --max_steps 5000 \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --metric_for_best_model eval_exact_match\
    --greater_is_better True \
    --task_names rte boolq wic mnli \
    --output_dir saved/ \
    --load_best_model_at_end \
    --disable_tqdm True --runs 1 --save_name mtl
```

This bash will fine-tune a `t5-small` model with `rte boolq wic mnli` datasets at the same time. 

The key configuration here are:

- `model_name_or_path` : representing the pretrained model name loaded from hugging face.
- `task_names` : representing the list of tasks we use to fine-tuning the LM.
- `output_dir` : representing the saved path for result.
- `downsample` : the number of samples that used for fine-tuning for each task.
- `max_steps`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps` are some useful configuration for fintuning, please adjust them when you need to.

### Results

The results will saved under the `results` floder as csv files.

## Run sampling script

Other than running a single expriment at a time, we provide a sampling script which is helpful to do multiple expriment for approximating the affinity matrix.

The corrisponding scripts includes `train_sample_tasks.py` and `train_sample_instructions.py`

### Sampling expriment for multitask fine-tuning

```bash 
python train_sample_tasks.py --model_name_or_path t5-base\
    --num_samples 1000 --min_task_num 2 --max_task_num 2\
    --max_steps 4000 --eval_steps 400 --save_steps 400 --runs 2\
    --task_set_name sample_test_dule_0_7 --save_name sample_test_dule_0_7 --device 0 \
    --target_task_group 0 7 --fp16 True
```

This script will sample `2` tasks from the two groups `0 7` to fine-tune a `t5-base` model. The groups are defined within the `train_sample_tasks.py`

The key configuration here are:
- `model_name_or_path` : representing the pretrained model name loaded from hugging face.
- `min_task_num`, `max_task_num`: the min/max number of tasks that will sampled from the given groups.
- `task_set_name` : The save name for the file that include all the completed tasks, whcih can be used to avoid the repeted running.
- `save_name` : The save name for results and checkpoints.


### Sampling expriment for multi_instruction fine-tuning

```bash
python train_sample_instructions_lora.py --num_tasks 10\
    --model_name_or_path t5-base --num_samples 50 --min_task_num 3 --max_task_num 3\
    --dataset rte --max_steps 1000 --eval_steps 200 --save_steps 200 --runs 2 --downsample 1000\
    --task_set_name sample_rte_lora --save_name sample_rte_lora --device 1 \
    --current_cluster_dir none --lora_rank 4 --lora_alpha 32
```

This script will sample `3` instructions for `rte` task to fine-tune the `t5-base` model with lora.

The key configuration here are:
- `model_name_or_path` : representing the pretrained model name loaded from hugging face.
- `min_task_num`, `max_task_num`: the min/max number of tasks that will sampled from the given groups.
- `task_set_name` : The save name for the file that include all the completed tasks, whcih can be used to avoid the repeted running.
- `save_name` : The save name for results and checkpoints.

