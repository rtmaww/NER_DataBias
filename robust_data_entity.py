
from transformers import AutoTokenizer
import os
import json
from data_process.data_utils import read_examples_from_file,get_labels
import pandas as pd

def aug_redudant_dataset_om(dataset_name,tokenizer_name="bert-base-cased",model_type = "bert"):
   
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data_path = f"{dataset_name}/PVI/split_by_pvi/{model_type}_om_supper.csv"
    
    om_supper_df = pd.read_csv(data_path,quotechar='"')

    label_type = get_labels(dataset_name,prefix=False)

    input_example = {}

    temp_examples = read_examples_from_file(f'{dataset_name}/data/',"train")
    for item in temp_examples:
        item.argu_mask=[1 for i in range(len(item.words))]
        input_example[int(item.sent_id)] = item
        
    
    
    for label in label_type:
        cur_label_df = om_supper_df[om_supper_df["label"]==label]
        
        for index,row in cur_label_df.iterrows():
            sent_id = row['sent_id']
            example = input_example[sent_id]
            oc_text = row["oc_text"]
            label = row["label"]
            oc_text_list = oc_text.split(' ')
            pos = oc_text_list.index(tokenizer.mask_token)
            
            if example['argu_mask'][pos] != 1:
                print(row)
                print(example['argu_mask'])
            assert("B-"+label == example['labels'][pos] and example['argu_mask'][pos] ==1)
            
            example['argu_mask'][pos] = 0
            pos+=1
            while pos < len(example['labels']) and example['labels'][pos] == "I-"+label:
                assert(example['argu_mask'][pos]==1)
                example['argu_mask'][pos] = 0
                pos+=1
            input_example[sent_id]=example
            
    out_dir = f'{dataset_name}/data/replace_om_bigger_Redundant'
    os.makedirs(out_dir,exist_ok=True)
    with open(f'{out_dir}/{model_type}_train.json','w',encoding='utf-8') as f:
        for sent_id in input_example.keys():
            instance=input_example[sent_id]
            json.dump({'sent_id':instance['sent_id'],'text':instance['words'],'label':instance['labels'],'label_mask':instance['label_mask'],'argu_mask':instance['argu_mask']},f)
            f.write('\n')
            
            
            
            
       
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default="conll2003")
    parser.add_argument("--tokenizer_path",type=str,default="bert-base-cased")
    parser.add_argument("--model_type",type=str,default="bert")
    args = parser.parse_args()
    aug_redudant_dataset_om(args.dataset_name,args.tokenizer_path,args.model_type)
    
    

        
   