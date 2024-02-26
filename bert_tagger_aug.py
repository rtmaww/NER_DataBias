
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import sys
import os
import random
import json
import numpy as np

import torch
from datasets import load_metric
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
print(os.getcwd())
print(sys.path)
from data_process.data_utils import convert_examples_to_features, get_labels, read_examples_from_file,read_pvi_file

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, AutoTokenizer,AutoConfig,AutoModel
from utils_augmentation import build_aug_examples

logger = logging.getLogger(__name__)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def cal_metric(preds,out_label_ids,labels,pad_token_label_id):
    preds = np.argmax(preds, axis=2)
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    label_map = {i: label for i, label in enumerate(labels)}
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    metric = load_metric("metric/seqeval.py")
    results=metric.compute(predictions=preds_list, references=out_label_list)
   
    return results

def train(args, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
  
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode= args.train_mode)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
   
   

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    args.logging_steps = int(len(train_dataset)/args.per_gpu_train_batch_size)
    print(f"logging_steps:{args.logging_steps}")
    #args.save_steps = args.logging_steps
    global_step = 0
    
    best_metric = 0.0
    best_dev_metric = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    cur_epoch=0
    for _ in train_iterator:
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            
            
            loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # model.zero_grad()
                optimizer.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if  args.evaluate_during_training :  # Only evaluate when single GPU otherwise metrics may not average well
                        
                        dev_results= evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")

                        if dev_results["overall_f1"] > best_dev_metric:
                            best_dev_metric = dev_results["overall_f1"]
                            output_dir = os.path.join(args.output_dir, "checkpoint-dev-best")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            tokenizer.save_pretrained(output_dir)
                            logger.info("Saving model checkpoint to %s", output_dir)
                       
                    
                if  args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        cur_epoch+=1

    return global_step, tr_loss / global_step

def fast_evaluate(args, ckpt_dir, config, tokenizer, labels,
                  mode, prefix='', model=None):
    if not model:
        model = AutoModel.from_pretrained(
            ckpt_dir,
            config=config
        )
        model.to(args.device)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    result = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode=mode, prefix=prefix)
    
    return result


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[0], outputs[1]
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
       
    results = cal_metric(preds,out_label_ids,labels,pad_token_label_id)
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir+mode+'.json')
        examples = read_examples_from_file(args.data_dir, mode)
        if 'train' in mode:

            change_examples = build_aug_examples(examples=examples,
                                                  aug_mode=args.aug_mode,
                                                  aug_rate=args.aug_rate,
                                                  aug_om_small_mode=args.aug_om_small_mode,
                                                  dataset_name=args.dataset_name,
                                                  om_small_json=args.om_small_json,
                                                  change_type=args.change_type
                                                  )
            
            
           
            examples=change_examples
            
            
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        if 'train' not in mode:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="conll2003/", type=str)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--dataset_name", default="conll2003", type=str)
    
    parser.add_argument("--output_dir", default="test", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_mode", default="OC_bigger_train_1", type=str)
    ## Other parameters
    parser.add_argument("--aug_mode", default="om_small_entity", type=str)
    parser.add_argument("--aug_rate", default=1, type=float)   
    parser.add_argument("--change_type", default="aug", type=str, help="replace or augmentation")      
    parser.add_argument("--om_small_file_name", type=str, default="OM_smaller_total_entity.json", help="For distant debugging.")
    parser.add_argument("--aug_om_small_mode", type=str, default="None", help="For distant debugging.")
    
            
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
   
    parser.add_argument("--gpu_id", type=int, default=1, help="For distant debugging.")
   
   
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
   
   
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
   
    args.om_small_json = f'{args.data_dir}/{args.om_small_file_name}'
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO )
    logger.warning(" device: %s, n_gpu: %s, 16-bits training: %s",
                    device, args.n_gpu, args.fp16)

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.dataset_name)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

   
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = AutoModel.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
    #if True:
        global_step, tr_loss = train(args, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    best_checkpoint = os.path.join(args.output_dir, "checkpoint-dev-best")
    if args.do_train and  not os.path.exists(best_checkpoint):
        os.makedirs(best_checkpoint,exist_ok=True)
        logger.info("Saving model checkpoint to %s", best_checkpoint)
       
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(best_checkpoint)
        tokenizer.save_pretrained(best_checkpoint)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(best_checkpoint, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = AutoTokenizer.from_pretrained(best_checkpoint, do_lower_case=args.do_lower_case)
        checkpoints = [best_checkpoint]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            result = fast_evaluate(args=args,ckpt_dir=checkpoint,config=config,tokenizer=tokenizer,labels=labels,mode='dev',prefix=global_step)
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        print(best_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(best_checkpoint, do_lower_case=args.do_lower_case)
        result = fast_evaluate(args=args,ckpt_dir=best_checkpoint,config=config,tokenizer=tokenizer,labels=labels,mode='test',prefix="best")
        oov_result= fast_evaluate(args=args,ckpt_dir=best_checkpoint,config=config,tokenizer=tokenizer,labels=labels,mode="trans_test",prefix="best")
        category_results= fast_evaluate(args=args,ckpt_dir=best_checkpoint,config=config,tokenizer=tokenizer,labels=labels,mode='category_test',prefix="best")
        
        log_results_item={'ori':result,'oov':oov_result,'category':category_results}
       
        # Save results
        output_test_results_file = os.path.join(best_checkpoint, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for item_name,item in log_results_item.items():
                for key in item.keys():
                    writer.write("{} = {}\n".format(f'{item_name}_{key}', str(item[key])))
        
    return results


if __name__ == "__main__":
    main()
