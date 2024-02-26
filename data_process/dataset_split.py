import json
import random
from data_transformation import generate_OC_OM
random.seed(3)

#split data n group for train and test
def data_split(data_path,output_path,split_number):
    with open(data_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        
    example_num = len(lines)
    print(f'total_dataset_count:{example_num}')
    total_set = set([i for i in range(example_num)])
    
    group_example_num = int(example_num/split_number)
    group_id_dict={key:[] for key in range(split_number)}
    for i in range(split_number-1):
        id_set = random.sample(total_set,group_example_num)
        group_id_dict[i] = list(id_set)
        total_set = total_set - set(id_set)
    group_id_dict[split_number-1] = list(total_set)
    
    
   
    for ind in range(split_number):
        train_id=[]
        test_id= group_id_dict[ind]
        for group_id,id_list in group_id_dict.items():
            if group_id != ind:
                train_id.extend(id_list)
        train_file = output_path + f"train_{ind}.json"
        test_file = output_path + f"test_{ind}.json"
        
        with open(train_file,'w',encoding='utf-8') as f:
                for line_ind in train_id:
                    f.write(lines[line_ind])

        with open(test_file,'w',encoding='utf-8') as f:
                for line_ind in test_id:
                    f.write(lines[line_ind])
                   
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default="conll2003")
    parser.add_argument("--data_path",type=str,default="conll2003/data/train.json")
    parser.add_argument("--output_path",type=str,default="conll2003/data/")
    parser.add_argument("--split_number",type=int, default= 5)
    parser.add_argument("--tokenizer_path",type=str,default="bert-base-cased")
    parser.add_argument("--tokenizer_name",type=str,default="bert")
    args = parser.parse_args()
    

   
    data_split(data_path=args.data_path,output_path=args.output_path,split_number=args.split_number)
    for test_id in range(args.split_number):
        generate_OC_OM(output_path=args.output_path,tokenizer_path=args.tokenizer_path,tokenizer_name=args.tokenizer_name,train_id=test_id)