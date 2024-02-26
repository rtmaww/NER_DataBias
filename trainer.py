# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random

import torch
import datasets
from datasets import load_dataset, load_metric
from EntityClassification import BertEntityClassification,RobertaEntityClassification
#from OmSeqClassification import OM_Seq_Classification
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    BertConfig,
    RobertaConfig,
    BertTokenizer,
    RobertaTokenizer,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    BertForSequenceClassification,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version

MODEL_CLASS={"bert":(BertConfig,BertTokenizer,BertEntityClassification)
           }
logger = logging.getLogger(__name__)

model_dict={'bert':['bert-base-cased']}

def parse_args():
   
    model_len={'OC':128,'OM':32,'null':32}
    epoch={'OC':10,'OM':6,'null':1}
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--gpu_id", type=int, default=0)
    parser.add_argument(
        "--train_id", type=int, default=0,help="inform the train/dev/test dataset name")
    parser.add_argument(
        "--data_dir", type=str, default="",help="inform the train/dev/test dataset dir")
    # parser.add_argument(
    #     "--orth", action="store_true" ,help="use the orth dataset")
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-cased",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    # ADDED
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bert-base-cased"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base-cased"
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=" ", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument("--model_mode", type=str, default="OC", help="select the model mode in ['OC','OM','null','std']")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="select the model mode in ['OC','OM','null','std']")
    args = parser.parse_args()
    if args.model_mode == 'null':
        args.num_train_epochs = epoch[args.model_mode]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

   
    
    
    out_model_name = args.model_type
  
    args.train_file = f"{args.data_dir}/train_{args.model_type}_{args.train_id}_{args.model_mode}.json"
    args.output_dir = f"{args.output_dir}/{out_model_name}_{args.train_id}_{args.model_mode}"
    print(f"train_file:{args.train_file}")
    args.validation_file = f"{args.data_dir}/test_{args.model_type}_{args.train_id}_{args.model_mode}.json"
    args.max_length = model_len[args.model_mode]
    
    
    # Sanity checks
    if  args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def get_dataloader(args,tokenizer):
    data_files = {}
    print(f"train-file:{args.train_file}")
    print(f"validation-file:{args.validation_file}")
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    print(f"loading the dataset completed ....")
    print(raw_datasets)
    print(len(raw_datasets['train']['label']))
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    
    print(f"label_list:{label_list}")
    label_to_id = {v: i for i, v in enumerate(label_list)}

    
    padding = "max_length" if args.pad_to_max_length else False

    
    def preprocess_function(examples):
        # Tokenize the texts
        texts = examples['text']
       
        result = tokenizer(texts, padding=padding,max_length=args.max_length, truncation=True)
        if "pos" in examples:
            result['ent_pos'] = examples["pos"]
        if "label" in examples:
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
        load_from_cache_file=False
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

   
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    
    return train_dataset,eval_dataset,label_list,label_to_id



def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    train_dataset,eval_dataset,label_list,label_to_id = get_dataloader(args,tokenizer)
    num_labels = len(label_list)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    if args.model_type == "bert":
      
        model = BertEntityClassification(config=config)


    elif args.model_type == 'roberta':
      
        model = RobertaEntityClassification(config=config,model_name = args.model_name_or_path)

    else:
        raise ValueError("args.model_name_or_path error，assert(it's in bert or roberta!!)")
       
    
    
    
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    
    if args.pad_to_max_length:
       
        data_collator = default_data_collator
    else:
      
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
   
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps < 0 or args.model_mode == "null":
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    metric = load_metric("metric/accuracy.py")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    train_range=max(1,args.num_train_epochs)
    best_metric = 0.0
    for epoch in range(train_range):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            total_loss += loss
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        train_metric = metric.compute()
        logger.info(f"epoch {epoch}   train_acc:{train_metric}   loss:{total_loss}")
        torch.cuda.empty_cache() # free memory
        model.eval()

        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}     eval_acc:{eval_metric}")
        if args.output_dir is not None and eval_metric['accuracy'] > best_metric:
            best_metric = eval_metric['accuracy']
            file_name = os.path.join(args.output_dir,"checkpoint-best")
            with open(f"{args.output_dir}/eval_results.txt",'w',encoding='utf-8') as f:
                f.write(f"epoch {epoch}     eval_acc:{eval_metric}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(file_name, save_function=accelerator.save)
            logger.info(f'save the epoch{epoch} checkpoint in {file_name}')


if __name__ == "__main__":
    main()
