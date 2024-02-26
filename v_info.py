from transformers import pipeline, AutoTokenizer,  default_data_collator,AutoConfig
from EntityClassification import BertEntityClassification,RobertaEntityClassification
import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from datasets import load_dataset,load_metric
import argparse
from torch.utils.data.dataloader import DataLoader
import csv

def get_Dataloader(args,tokenizer,config,data_fn):
    data_files={}
    data_files['test'] = data_fn
    extension = 'json'
    raw_datasets = load_dataset(extension, data_files=data_files)
    label_to_id = config.label2id
    padding = "max_length"
    def preprocess_function(examples):
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
        remove_columns=raw_datasets["test"].column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = processed_datasets["test"]
    return test_dataset
    


def split_pvi_by_labels(pvi_out_path,train_id=0,mode='OC'):
    file_name = pvi_out_path
    file_dir = pvi_out_path.split('/')[0]
    label_ind_dict={'label':4}
    file_reader = open(file_name)
    reader = list(csv.reader(file_reader))
    title = reader[0]
    content = reader[1:]
    labels = [item[label_ind_dict['label']] for item in content]
    label_type = set(labels)
    title = ','.join(title)
    output_dict = {f'{label}_output':title for label in label_type}
    for i,label in enumerate(labels):
        content[i][3]=content[i][3].replace(',',';')
        out_content = '\n'+','.join(content[i])
        output_dict[f'{label}_output'] += out_content
    out_dir = os.path.join(file_dir,"split_by_label",f'{train_id}',f'{mode}')
    os.makedirs(out_dir,exist_ok=True)
    for label in label_type:
        out_file_name = out_dir + f'/{label}_test.csv'
        with open(out_file_name,'w',encoding='utf-8') as f:
            f.write(output_dict[f'{label}_output'])



def v_entropy(args,data_fn, model_path, tokenizer, input_key='sentence1', batch_size=100):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
    print(device)
    
    if args.model_type == "bert":
        model = BertEntityClassification.from_pretrained(model_path)
    elif args.model_type == "roberta":
        model = RobertaEntityClassification.from_pretrained(model_path)
        
    config = AutoConfig.from_pretrained(model_path)
    tokenizer= AutoTokenizer.from_pretrained(tokenizer)
    
        
    test_dataset = get_Dataloader(args,tokenizer=tokenizer,config = config,data_fn=data_fn)
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=32)
    model.to(device)
    model.eval()

    entropies = []
    correct =[]
    predicted_labels=[]
    probability = []
    metric = load_metric("metric/accuracy.py")
    id_to_label = config.id2label
    for batch in tqdm(test_dataloader):
        batch={k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        correct_label = batch['labels']

        predictions = outputs.logits.argmax(dim=-1)
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(outputs.logits)
        for i in range(len(batch['labels'])):
            predicted_label = predictions[i].cpu().numpy().tolist()
            gold_label = correct_label[i].cpu().numpy().tolist()
            prob = probs[i][gold_label].detach().cpu().numpy().tolist()
            probability.append(prob)
            entropies.append(-1 * np.log2(prob))
            predicted_labels.append(id_to_label[predicted_label])
            correct.append(predicted_label == gold_label)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )

    eval_metric = metric.compute()
    print(f"acc:{eval_metric}")

    torch.cuda.empty_cache()

    return entropies, correct, predicted_labels,probability

def json_to_csv(file_name):
    with open(f'{file_name}.json', 'r',encoding ='utf-8') as f:
        lines = f.readlines()
        
    json_lines=[]
    for line in lines:
        line = json.loads(line)
        json_lines.append(line)
        
    output = '\t'.join([*json_lines[0]])
    print(output)
     
    for obj in json_lines:
          
        if 'OC' in file_name:
            output += f'\n{obj["doc_id"]}\t{obj["sent_id"]}\t{obj["text"]}\t{obj["label"]}\t{obj["pos"]}'
        elif 'OM' in file_name:
            output += f'\n{obj["doc_id"]}\t{obj["sent_id"]}\t{obj["text"]}\t{obj["label"]}'
        else:
            print(f'{file_name} error!!')
       
    csv_file_name = f'{file_name}.csv'
    with open(csv_file_name, 'w') as f:
        f.write(output)
    return csv_file_name
    

def v_info(args,data_fn, model, null_data_fn, null_model, tokenizer, out_fn="", input_key='text'):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a 
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        null_data: path to null data (column specified by input_key should have
            null data)
        null_model: path to saved model trained on null data
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        out_fn: where to saved 
        input_key: column name of X variable in data_fn 

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional 
        columns specified above.
    """
    temp_file = data_fn.split('.')
    real_file = json_to_csv(temp_file[0])
    print(real_file)
    data = pd.read_csv(real_file,sep='\t',quoting=csv.QUOTE_NONE)
    data['H_yb'], _, _ ,_= v_entropy(args,null_data_fn, null_model, tokenizer, input_key=input_key) 
    data['H_yx'], data['correct_yx'], data['predicted_label'],data['prob'] = v_entropy(args,data_fn, model, tokenizer, input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']

    if out_fn:
        data.to_csv(out_fn)

    return data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_id',default = 0,type =int)
    parser.add_argument('--tokenizer_name',default = 'bert-base-cased',type =str)
    parser.add_argument('--model_type',default = 'bert',type =str)
    parser.add_argument('--dataset_name',default = "ontonotes",type =str)
    parser.add_argument('--data_dir', default='ontonotes/data/',  type=str)
    parser.add_argument('--model_dir', default='ontonotes/model',  type=str)
    parser.add_argument('--gpu_id', default=0,  type=int)
   

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    DATA_DIR = args.data_dir
    MODEL_DIR = args.model_dir
    pvi_path = os.path.join(args.dataset_name,'PVI')
    os.makedirs(pvi_path, exist_ok=True)
    

    model_name = args.model_type
   
    oc_model_path = f"{model_name}_{args.train_id}_OC"
    om_model_path = f"{model_name}_{args.train_id}_OM"
    null_model_path = f"{model_name}_{args.train_id}_null"
    experiments = [
        (f"test_{args.model_type}_{args.train_id}_OC.json", oc_model_path,"best"),
        (f"test_{args.model_type}_{args.train_id}_OM.json", om_model_path,"best"),
    ]
    for side_data, side_model,ckpt_ind in experiments:
        if 'OM' in side_data:
            args.max_length = 32
        elif 'OC' in side_data:
            args.max_length = 128
        else:
            print('error')
            exit()
            
        pvi_out_path = f"{pvi_path}/{side_model}_test.csv"
        v_info(args,f"{DATA_DIR}/{side_data}", f"{MODEL_DIR}/{side_model}/checkpoint-{ckpt_ind}", f"{DATA_DIR}/test_{args.model_type}_{args.train_id}_null.json", f"{MODEL_DIR}/{null_model_path}/checkpoint-best", args.tokenizer_name,
            out_fn=pvi_out_path)
        mode = side_model.split('_')[-1]
        print(f'mode:{mode}')
        split_pvi_by_labels(pvi_out_path,train_id=args.train_id,mode=mode)
