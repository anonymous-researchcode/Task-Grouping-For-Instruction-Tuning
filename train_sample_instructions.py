import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
import argparse
import numpy as np
import shutil

def main(args):
    task_list = np.arange(args.num_tasks)
    task_list = [str(task) for task in task_list]

    target_tasks = args.target_tasks
    other_tasks = [task for task in task_list if task not in target_tasks]

    num_samples = args.num_samples
    max_task_num = args.max_task_num
    min_task_num = args.min_task_num
    for _ in range(num_samples):
        # create a set of trained task combinations
        sampled_task_dir = os.path.join("./sampled_tasks", "{}.txt".format(args.task_set_name))
        if not os.path.exists(sampled_task_dir):
            f = open(sampled_task_dir, "w")
            f.close()
            
        with open(sampled_task_dir, "r") as f:
            sampled_tasks = set()
            for line in f.readlines():
                sampled_tasks.add(line.rstrip("\n"))
            # print(sampled_tasks)

        # train on a new task combination
        with open(sampled_task_dir, "a") as f:
            if len(target_tasks) == 0:
                tmp_other_task_num = np.random.randint(
                    low=min_task_num, 
                    high=max_task_num+1
                )
                tmp_sampled_other_tasks = np.random.choice(other_tasks, size=tmp_other_task_num,replace=False)
                
                tmp_sampled_tasks = tmp_sampled_other_tasks
                tmp_sampled_tasks.sort()
                tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            else:
                tmp_target_task_num = np.random.randint(low=1, high=len(target_tasks)+1)
                tmp_sampled_target_tasks = np.random.choice(target_tasks, size=tmp_target_task_num, replace=False)

                tmp_other_task_num = np.random.randint(
                    low=max(min_task_num-tmp_target_task_num, 0), 
                    high=max_task_num-tmp_target_task_num+1
                )
                tmp_sampled_other_tasks = np.random.choice(other_tasks, size=tmp_other_task_num,replace=False)
                
                tmp_sampled_tasks = np.concatenate([tmp_sampled_target_tasks, tmp_sampled_other_tasks])
                tmp_sampled_tasks.sort()
                tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            
            if tmp_sampled_tasks in sampled_tasks:
                continue
            print(tmp_sampled_tasks)
            
            os.system("CUDA_VISIBLE_DEVICES={} python train_multi_instruction_v3.py \
                        --do_train \
                        --do_eval \
                        --predict_with_generate \
                        --model_name_or_path {} \
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
                        --learning_rate {} \
                        --lr_scheduler_type constant \
                        --max_steps {} \
                        --warmup_steps 0 \
                        --logging_strategy steps \
                        --logging_steps 500 \
                        --evaluation_strategy steps \
                        --eval_steps {} \
                        --save_strategy steps \
                        --save_steps {} \
                        --metric_for_best_model eval_exact_match\
                        --greater_is_better True \
                        --downsample {} \
                        --task_name {} \
                        --template_idx {} \
                        --output_dir saved/ \
                        --load_best_model_at_end \
                        --disable_tqdm True \
                        --runs {}\
                        --save_name {}".format(
                            args.device, args.model_name_or_path, args.lr, args.max_steps, args.eval_steps, args.save_steps,
                            args.downsample, args.dataset, tmp_sampled_tasks, args.runs, args.save_name
            ))
            instruction_idxs = "[" + ", ".join(tmp_sampled_tasks.split(" ")) + "]"
            output_dir = os.path.join("saved", "{}_{}_{}".format(args.dataset, instruction_idxs, args.model_name_or_path))
            # delete the output_dir
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            sampled_tasks.add(tmp_sampled_tasks)
            f.write(tmp_sampled_tasks + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--target_tasks", nargs='+', type=str, default=[])

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--min_task_num", type=int, default=3)
    parser.add_argument("--max_task_num", type=int, default=3)

    parser.add_argument("--dataset", type=str, default="rte")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--downsample", type=int, default=500)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--task_set_name", type=str, default="sampled_tasks")
    parser.add_argument("--save_name", type=str, default="sampled_tasks")

    args = parser.parse_args()
    main(args)