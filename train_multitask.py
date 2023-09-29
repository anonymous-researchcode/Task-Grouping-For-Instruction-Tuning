"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
import datasets
import nltk
import numpy as np
from datasets.utils import disable_progress_bar
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from data_loader import Seq2SeqMultitaskCollator, Seq2SeqMultiInstructionCollator
from trainer import NITrainer, DecoderOnlyTrainer, DenserEvalCallback
from compute_metrics import compute_metrics, compute_grouped_metrics
from promptsource.templates import DatasetTemplates, Template
from task_constants import task_to_benchmark, task_to_instruction_template, task_is_generative_task
from utils.adjustment import add_adjustment_term, split_gpt_self_attention
from utils.template_utils import asssign_template, apply_template
from copy import deepcopy
from transformers.optimization import Adafactor, AdafactorSchedule
from utils.util import add_result_to_csv
import copy
import os
os.environ["WANDB_DISABLED"] = "true"

disable_progress_bar()
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    New arguments
    """
    task_names : str = field(
        default=None, 
        metadata={"help": "The name of the task to train on.", "nargs": "+",}
    )
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    
    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    train_adjustment: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, train the adjustment terms."}
    )
    freeze_module : Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, freeze the pretrained parameters."}
    )
    module_name : Optional[str] = field(
        default="query",
        metadata={"help": "If specified, finetune the pretrained model."}
    )
    freeze_layer : Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, freeze the pretrained parameters."}
    )
    layer_index : Optional[int] = field(
        default=0,
        metadata={"help": "If specified, finetune the pretrained model."}
    )
    runs : Optional[int] = field(
        default=3,
        metadata={"help": "The number of random seeds"}
    )
    downsample : Optional[int] = field(
        default=-1,
        metadata={"help": "If specified, downsample the training set."}
    )
    use_Adafactor_optm: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, use Adafactor optimizer"}
    )
    is_generative_task: Optional[bool] = field(
        default=False,
        metadata={"help": "If the task is a generative task"}
    )
    save_name: Optional[str] = field(
        default="test",
        metadata={"help": "The name of the model to save."}
    )
    train_multitask: Optional[bool] = field(
        default=True,
        metadata={"help": "If specified, train the model on multiple tasks."}
    )

def load_multitask_datasets(data_args, training_args):
    # split dataset into train, validation, test
    task_names = data_args.task_names
    multitask_train_dataset, multitask_valid_dataset, multitask_test_dataset = [], [], []
    task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets, task_to_templates = {}, {}, {}, {}

    for task_name in task_names:
        benchmark_name = task_to_benchmark[task_name] # Test set does not have labels

        if benchmark_name is not None:
            print(f"loading dataset {benchmark_name}/{task_name}....")
            raw_datasets = load_dataset(benchmark_name, task_name)
            dataset_templates = DatasetTemplates(benchmark_name, task_name)
            print(f"{benchmark_name}/{task_name} loading completed!")
        else:
            print(f"loading dataset {task_name}....")
            raw_datasets = load_dataset(task_name)
            dataset_templates = DatasetTemplates(task_name)
            print(f"{task_name} loading completed!")
        
        keys = list(dataset_templates.name_to_id_mapping.keys())
        if task_is_generative_task[task_name]:
            templates = [dataset_templates[key] for key in keys if 'ROUGE' in dataset_templates[key].metadata.metrics]
        else:
            templates = [dataset_templates[key] for key in keys]

        i = 0
        template = templates[i]
        while not template.answer_choices and (i + 1) < len(templates):
            i += 1
            template = templates[i]

        # unifying labels space
        template = Template(name="defined", jinja=task_to_instruction_template[task_name], reference="", answer_choices = template.answer_choices)

        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        valid_name = "validation_matched" if task_name == "mnli" else "validation"
        test_name = "test_matched" if task_name == "mnli" else "test"
        if training_args.do_eval:
            if valid_name not in raw_datasets:
                print("No validation dataset, using test set")
                if test_name not in raw_datasets:
                    raise ValueError("--do_eval requires a validation dataset")
                eval_dataset = raw_datasets[test_name]
            else:
                eval_dataset = raw_datasets[valid_name] 
        else:
            eval_dataset = []

        if training_args.do_predict:
            if test_name not in raw_datasets:
                print("No test dataset, using validation set")
                if valid_name not in raw_datasets:
                    raise ValueError("--do_predict requires a test dataset")
                predict_dataset = raw_datasets[valid_name]
            else:
                predict_dataset = raw_datasets[test_name] 
        else:
            predict_dataset = []

        # downsample training set
        if training_args.downsample > 0 and len(train_dataset) > training_args.downsample:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(train_dataset), training_args.downsample, replace=False)
            train_dataset = train_dataset.select(indices)

        if (training_args.downsample > 0) and (len(eval_dataset) > training_args.downsample):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(eval_dataset), training_args.downsample, replace=False)
            eval_dataset = eval_dataset.select(indices)

        if (training_args.downsample > 0) and (len(predict_dataset) > training_args.downsample):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(predict_dataset), training_args.downsample, replace=False)
            predict_dataset = predict_dataset.select(indices)

        print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))

        # Prepare validation and test dataset
        column_names = train_dataset.column_names
        if "input" in column_names:
            column_names.remove("input")
            train_dataset = train_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
            train_dataset.remove_columns(["old_input"])
        else:
            train_dataset = train_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
        multitask_train_dataset.append(train_dataset)
        task_to_train_datasets[task_name] =  copy.deepcopy(train_dataset)
        task_to_templates[task_name] = template
        

        if training_args.do_eval:
            column_names = eval_dataset.column_names
            if "input" in column_names:
                column_names.remove("input")
                eval_dataset = eval_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
                eval_dataset.remove_columns(["old_input"])
            else:
                eval_dataset = eval_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
            multitask_valid_dataset.append(eval_dataset)
            task_to_valid_datasets[task_name] = copy.deepcopy(eval_dataset)

        if training_args.do_predict:
            column_names = predict_dataset.column_names
            if "input" in column_names:
                column_names.remove("input")
                predict_dataset = predict_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
                predict_dataset.remove_columns(["old_input"])
            else:
                predict_dataset = predict_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
            multitask_test_dataset.append(predict_dataset)
            task_to_test_datasets[task_name] =  copy.deepcopy(predict_dataset)

    multitask_train_dataset = datasets.concatenate_datasets(multitask_train_dataset)
    batch_size = training_args.per_device_train_batch_size 
    shuffle_indices = np.arange(len(multitask_train_dataset) - len(multitask_train_dataset) % batch_size)
    shuffle_indices = np.reshape(shuffle_indices, (-1, batch_size))
    rng = np.random.default_rng(42)
    rng.shuffle(shuffle_indices, axis=0)
    shuffle_indices = np.reshape(shuffle_indices, (-1))
    # multitask_train_dataset = multitask_train_dataset.shuffle(seed=42)
    # multitask_train_dataset = multitask_train_dataset.flatten_indices()
    multitask_train_dataset = multitask_train_dataset.select(shuffle_indices)
    print("multitask train dataset size: {}".format(len(multitask_train_dataset)))

    if training_args.do_eval:
        multitask_valid_dataset = datasets.concatenate_datasets(multitask_valid_dataset)
        print("multitask validation dataset size: {}".format(len(multitask_valid_dataset)))
    
    if training_args.do_predict:
        multitask_test_dataset = datasets.concatenate_datasets(multitask_test_dataset)
        print("multitask test dataset size: {}".format(len(multitask_test_dataset)))
    
    return multitask_train_dataset, multitask_valid_dataset, multitask_test_dataset, \
        task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets, task_to_templates

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NITrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
            "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-base"
        ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    task_name_str = "_".join(data_args.task_names)[:100]
    training_args.output_dir = os.path.join(training_args.output_dir,
                     "{}_{}".format(task_name_str, model_args.model_name_or_path))

    results = {}
    logging_metrics = ["loss", "accuracy", "exact_match", "rouge1", "rougeL", "rouge2"]
    for run in range(training_args.runs):

        # Set seed before initializing model.
        seed = np.random.randint(0, 1e6)
        set_seed(seed)
        training_args.seed = seed

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if model_args.model_name_or_path in [
                "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
                "google/t5-v1_1-base"
            ]:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif model_args.model_name_or_path in [
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        ]:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

            for name, module in model.named_modules():
                if "c_attn" in name:
                    split_gpt_self_attention(module, "weight")

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


        if training_args.freeze_module:
            if model_args.model_name_or_path in [
                "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-base"
            ]:
                module_name_to_t5_param_key = {
                    "embedding": "shared",
                    "query": ".q",
                    "key": ".k",
                    "value": ".v",
                    "output": ".o",
                    "dense_in": ".wi",
                    "dense_out": ".wo",
                }
                module_name = module_name_to_t5_param_key[training_args.module_name]
                for name, param in model.named_parameters():
                    if module_name in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif model_args.model_name_or_path in [
                "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
            ]:
                module_name_to_t5_param_key = {
                    "embedding": "wte",
                    "query": "weight_q",
                    "key": "weight_k",
                    "value": "weight_v",
                    "output": "attn.c_proj.weight",
                    "dense_in": "mlp.c_fc.weight",
                    "dense_out": "mlp.c_proj.weight",
                }
                module_name = module_name_to_t5_param_key[training_args.module_name]
                for name, param in model.named_parameters():
                    if (module_name in name) or ("bias" in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        elif training_args.freeze_layer:
            if model_args.model_name_or_path in [
                "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-base"
            ]:
                encoder_layers = len(model.encoder.block)
                layer_idx = training_args.layer_index % encoder_layers
                use_encoder = training_args.layer_index < encoder_layers
                module_name = "encoder" if use_encoder else "decoder"
                module_name = "{}.block.{}.".format(module_name, layer_idx)
                for name, param in model.named_parameters():
                    if module_name in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif model_args.model_name_or_path in [
                "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
            ]:
                layer_idx = training_args.layer_index
                module_name = "transformer.h.{}.".format(layer_idx)
                for name, param in model.named_parameters():
                    if module_name in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        if training_args.train_adjustment:
            keys = [
                ".q", ".k", ".v",
                ".o", ".wi", ".wo",
                ".layer_norm"
            ]

            for name, module in model.named_modules():
                for key in keys:
                    if key in name:
                        add_adjustment_term(module, "weight")

            # make pretrained weights untrainable
            for name, param in model.named_parameters():
                if "pretrained" in name:
                    param.requires_grad = False

        # if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        #     if isinstance(tokenizer, MBartTokenizer):
        #         model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        #     else:
        #         model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)
        # if model.config.decoder_start_token_id is None:
        #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                    f"to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )

        if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
            assert (
                data_args.lang is not None
            ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

            tokenizer.src_lang = data_args.lang
            tokenizer.tgt_lang = data_args.lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
            forced_bos_token_id = (
                tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
            )
            model.config.forced_bos_token_id = forced_bos_token_id


        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        multitask_train_dataset, multitask_valid_dataset, multitask_test_dataset, \
        task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets, \
        task_to_templates = load_multitask_datasets(data_args, training_args)
        
        # Metric
        def compute_ni_metrics(dataset, preds, save_prefix=None):
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            references = [[e['output']] for e in dataset] # examples has been processed
            result = compute_metrics(predictions=decoded_preds, references=references)
            
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            
            return result

        if model_args.model_name_or_path in [
                "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-base"
            ]:
            # Data collator
            label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
            data_collator = Seq2SeqMultitaskCollator(
                tokenizer, model,
                padding="max_length" if data_args.pad_to_max_length else "longest",
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8,
                add_task_name=data_args.add_task_name,
                add_task_definition=data_args.add_task_definition,
                num_pos_examples=data_args.num_pos_examples,
                num_neg_examples=data_args.num_neg_examples,
                add_explanation=data_args.add_explanation,
                tk_instruct=data_args.tk_instruct
            )
            # we don't want to remove unused columns because we will prepare each batch during training, 
            # and some of the information will aslo be used in evaluation.
            training_args.remove_unused_columns = False 

            # switch to different optimizer if needed
            if training_args.use_Adafactor_optm:
                optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=False,
                                      lr=training_args.learning_rate)
                lr_scheduler = AdafactorSchedule(optimizer, initial_lr=training_args.learning_rate)
            else:
                optimizer, lr_scheduler = None, None

            # Initialize our Trainer
            trainer = NITrainer(
                model=model,
                args=training_args,
                train_dataset=multitask_train_dataset if training_args.do_train else None,
                eval_dataset=multitask_valid_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
                callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
                optimizers=(optimizer, lr_scheduler)
            )

        all_metrics = {"run_name": training_args.run_name}

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(multitask_train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(multitask_train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            all_metrics.update(metrics)

        # Evaluation
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.max_target_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            ''' Evaluate for all possible templates '''
            for task_name in data_args.task_names:
                eval_dataset = task_to_valid_datasets[task_name]

                trainer.eval_dataset = eval_dataset
                metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
                max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                new_metrics = {}
                for key, val in metrics.items():
                    if key.startswith("eval"):
                        key = key.replace("eval", f"eval_{task_name}")
                        new_metrics[key] = val

                trainer.log_metrics(f"eval_{task_name}", metrics)
                trainer.save_metrics(f"eval_{task_name}", metrics)

                all_metrics.update(new_metrics)

        if training_args.do_predict:
            logger.info("*** Predict ***")

            for task_name in data_args.task_names:
                predict_dataset = task_to_test_datasets[task_name]

                predict_results = trainer.predict(
                    predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
                )
                metrics = predict_results.metrics
                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
                )
                metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

                new_metrics = {}
                for key, val in metrics.items():
                    if key.startswith("predict"):
                        key = key.replace("predict", f"predict_{task_name}")
                        new_metrics[key] = val

                trainer.log(metrics)
                trainer.log_metrics(f"predict_{task_name}", metrics)
                trainer.save_metrics(f"predict_{task_name}", metrics)

                all_metrics.update(metrics)

        for key, val in all_metrics.items():
            for metric_name in logging_metrics:
                if (metric_name in key) and (key not in results):
                    results[key] = [val, ]
                elif (metric_name in key) and (key in results):
                    results[key].append(val)

    for key, val in results.items():
        print("{}: {:.4f} +/- {:.4f}".format(key, np.mean(val), np.std(val)))

    # save results into .csv
    file_dir = os.path.join("./results/", training_args.save_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    for task_name in data_args.task_names:
        # save validation results
        if training_args.do_eval:
            result_datapoint = {
                "Task": task_name, 
                "Trained on": data_args.task_names,
            }
            for key, vals in results.items():
                if f"eval_{task_name}" in key:
                    key = key.replace(f"eval_{task_name}", "eval")
                    result_datapoint[key] = np.mean(vals)
                    result_datapoint[key+"_std"] = np.std(vals)
            file_name = os.path.join(file_dir, "{}_valid.csv".format(training_args.save_name))
            add_result_to_csv(result_datapoint, file_name)

        # save test results
        if training_args.do_predict:
            result_datapoint = {
                "Task": task_name, 
                "Trained on": data_args.task_names,
            }
            for key, vals in results.items():
                if f"predict_{task_name}_" in key:
                    key = key.replace(f"predict_{task_name}", "predict")
                    result_datapoint[key] = np.mean(vals)
                    result_datapoint[key+"_std"] = np.std(vals)
            file_name = os.path.join(file_dir, "{}_test.csv".format(training_args.save_name))
            add_result_to_csv(result_datapoint, file_name)

if __name__ == "__main__":
    main()