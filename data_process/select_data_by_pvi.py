
import os
import pandas as pd
from transformers import AutoTokenizer
import json
from data_utils import InputExample,mk_dir,get_labels,read_examples_from_file,stas_data_distribution,Entity
import argparse



def PVI_classification(file_dir,split_number=1,model_type="bert"):
   
    # theta={'PERSON':0.8,'EVENT':4,'FAC':4,'LANGUAGE':4,'LAW':4,'ORG':1,'PRODUCT':4,'WORK_OF_ART':4,'NORP':1,'LOC':4,'GPE':1}
    theta = {'PER':0.8,'ORG':1.0,'LOC':1,'MISC':2}
    #theta={'PERSON':0.7,'EVENT':5.5,'FAC':5.3,'LANGUAGE':6.8,'LAW':7,'ORG':1.1,'PRODUCT':5.6,'WORK_OF_ART':5,'NORP':2.1,'LOC':4.26,'GPE':0.7}
   
    
    dataset_name = file_dir.split('/')[0]
    label_type = get_labels(dataset_name,prefix=False)
    
    for train_id in range(split_number):
        
        om_file_path = file_dir+f'/{model_type}_{train_id}_OM_test.csv'
        oc_file_path = file_dir+f'/{model_type}_{train_id}_OC_test.csv'
        
        om_df = pd.read_csv(om_file_path,quotechar='"')
        oc_df = pd.read_csv(oc_file_path,quotechar='"')
        om_df.index.name = "index"
        oc_df.index.name = "index"
        df = oc_df.merge(om_df,on='index',how='left',suffixes=('_oc','_om'))
        print(df.columns)
        
        select_column = ['doc_id_om','sent_id_om','text_oc','text_om','label_om','correct_yx_oc','correct_yx_om','PVI_differ','PVI_om','PVI_oc',]
        rename_dict = {"doc_id_om":'doc_id','sent_id_om':'sent_id','text_oc':'oc_text','text_om':'om_text',\
            'label_om':'label','correct_yx_oc':'oc_true',"correct_yx_om":'om_true',"PVI_differ":'pvi_differ',"PVI_om":'om_pvi',"PVI_oc":'oc_pvi',}
        
        out_dir = os.path.join(file_dir,'split_by_pvi')
        split_by_label_dir = os.path.join(out_dir,'label_wise')
        mk_dir(out_dir)
        mk_dir(split_by_label_dir)
        
        theta_list = [theta[item] for item in df['label_om']]
        om_supper = df[df['PVI_om'] > theta_list]
        om_supper["PVI_differ"] = om_supper["PVI_om"] - om_supper["PVI_oc"]
        om_supper = om_supper[select_column].rename(columns=rename_dict)
        om_supper = om_supper.sort_values(by='pvi_differ',ascending=False)
        om_supper.to_json(f'{out_dir}/{model_type}_om_supper.json',orient='records')
        om_supper.to_csv(f'{out_dir}/{model_type}_om_supper.csv')
        
        
        oc_bigger = df[df['PVI_oc'] > df['PVI_om']]
        oc_bigger["PVI_differ"] = oc_bigger["PVI_oc"] - oc_bigger["PVI_om"]
        oc_bigger = oc_bigger[select_column].rename(columns=rename_dict)
        oc_bigger = oc_bigger.sort_values(by='pvi_differ',ascending=False)
        oc_bigger.to_json(f'{out_dir}/{model_type}_oc_bigger.json',orient='records')
        
        
        om_bigger = df[df['PVI_om'] > df['PVI_oc']]
        om_bigger["PVI_differ"] = om_bigger["PVI_om"] - om_bigger["PVI_oc"]
        om_bigger = om_bigger[select_column].rename(columns=rename_dict)
        om_bigger = om_bigger.sort_values(by='pvi_differ',ascending=False)
        om_bigger.to_json(f'{out_dir}/{model_type}_om_bigger.json',orient='records')
        
        
        for label in label_type:
            oc_bigger_by_label = oc_bigger[oc_bigger['label']==label]
            om_bigger_by_label = om_bigger[om_bigger['label']==label]
            
           
            oc_bigger_by_label.to_json(f'{split_by_label_dir}/{model_type}_{label}_oc.json',orient='records')
            om_bigger_by_label.to_json(f'{split_by_label_dir}/{model_type}_{label}_om.json',orient='records')
            
def duplicate_removal(dataset_name, label_wise=False,tokenizer_path="bert-base-cased"):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

   
    # raw_examples = read_examples_from_file(f'{dataset_name}/data',"train")
    # raw_input_example = {}
    # for example in raw_examples:
    #     raw_input_example[example["sent_id"]] = example
    raw_data_path = f"{dataset_name}/data/train.json"
    with open(raw_data_path,'r',encoding='utf-8') as f:
            instances = f.readlines()
    raw_input_example = {}
    for instance in instances:
        instance = json.loads(instance)
        label = instance['label']
        label_mask=[0 if label[i]=='O' else 1  for i in range(len(label))]
        raw_input_example[instance['sent_id']] = InputExample(instance['sent_id'],instance['text'],instance['label'],label_mask)
    
    

    if label_wise:
        folder_path = f'{dataset_name}/PVI/split_by_pvi/label_wise/'
    else:
        folder_path = f'{dataset_name}/PVI/split_by_pvi/'

    file_list = [f for f in os.listdir(folder_path) if os.path.join(folder_path, f).split('.')[-1]== "json"]

    print(file_list)
    import copy
    
    for file_ind,file_name in enumerate(file_list):
        
        sent_dict={}
        input_example = copy.deepcopy(raw_input_example)
        cur_file_name = f"{folder_path}/{file_name}"
        with open(cur_file_name) as f:
            lines = json.load(f)
            
        for line in lines:
            sent_id = line["sent_id"]
            if sent_id not in sent_dict:
                sent_dict[sent_id]=[]
            sent_dict[sent_id].append(line)
            
        for sent_id,sent_list in sent_dict.items():
            example = input_example[sent_id]
    
            for sentence in sent_list:
                OC_text = sentence['oc_text'].split(" ")
                label = sentence["label"]
                for pos ,item in enumerate(OC_text):
                    if item == tokenizer.mask_token:
                        break
                if example['label_mask'][pos] !=1:
                    print(sent_list)
                    print(example['label_mask'])
                    print(example['label'])
                
                assert("B-"+label == example['labels'][pos] and example['label_mask'][pos] ==1)
                example['label_mask'][pos] = 0
                pos+=1
                while pos < len(example['labels']) and example['labels'][pos] == "I-"+label:
                    assert(example['label_mask'][pos] == 1)
                    example['label_mask'][pos] = 0
                    pos+=1
                input_example[sent_id]=example
            
      
        with open(f'{folder_path}/{file_name}_new.json','w',encoding='utf-8') as f:
            for sent_id in sent_dict.keys():
                instance=input_example[sent_id]
                json.dump({'sent_id':instance['sent_id'],'text':instance['words'],'label':instance['labels'],'label_mask':instance['label_mask']},f)
                f.write('\n')


def get_OC_OM(dataset_name,split_number,tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    raw_examples_dict={}
    ori_examples = read_examples_from_file(f'{dataset_name}/data','train')
    for example in ori_examples:
        raw_examples_dict[example.sent_id] = example
    label_type = get_labels(dataset_name,prefix=False)
    print(label_type)
    
    oc_dict = {label:{} for label in label_type}
    om_dict ={label:{} for label in label_type}
  
    for split_ind in range(split_number):
        oc_file_name = f'{dataset_name}/PVI/{model_type}_{split_ind}_OC_test.csv'
        om_file_name = f'{dataset_name}/PVI/{model_type}_{split_ind}_OM_test.csv'
        oc_df = pd.read_csv(oc_file_name,quotechar='"')
        om_df = pd.read_csv(om_file_name,quotechar='"')
        om_df.index.name = "index"
        oc_df.index.name = "index"
        df = oc_df.merge(om_df,on='index',how='left',suffixes=('_oc','_om'))


        for index,row in df.iterrows():
            

            sent_id = row["sent_id_oc"]
            cur_label = row["label_om"]
            ori_example = raw_examples_dict[sent_id]

            oc_text_list = row["text_oc"].split()
            om_text_list = row["text_om"].split()
           
            
            mask_index = oc_text_list.index(tokenizer.mask_token)
            new_label = ori_example.labels[:mask_index] + ['<split>'] + ori_example.labels[mask_index+len(om_text_list):]

            label_mask_pre = [0 if element=='O' else 1 for element in ori_example.labels[0:mask_index] ]
            label_mask_post= [0 if element=='O' else 1 for element in ori_example.labels[mask_index+len(om_text_list):] ]
            new_label_mask = label_mask_pre+['<split>']+label_mask_post
            assert(len(oc_text_list)== len(new_label))
            assert(len(new_label)==len(new_label_mask))


            oc = row["text_oc"].replace(tokenizer.mask_token,'<split>')
            om = '<split>'.join(row["text_om"].split())

            if index <5:
                print(f'oc:{oc}')
                print(f'om:{om}')
            
            om_pvi = float(row["PVI_om"])
            oc_pvi = float(row["PVI_oc"])

            if oc not in oc_dict[cur_label]:
                oc_dict[cur_label] [oc] = {'pvi':oc_pvi,'label':new_label,'label_mask':new_label_mask}
            
            if om not in om_dict[cur_label]:
                om_dict[cur_label] [om] = om_pvi
                
            
    with open(f'{dataset_name}/data/oc_pvi.json','w',encoding='utf-8') as f:
        json.dump(oc_dict,f)
    
    with open(f'{dataset_name}/data/om_pvi.json','w',encoding='utf-8') as f:
        json.dump(om_dict,f)

    
def get_om_small_data(dataset_name,split_number,tokenizer_path):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    om_small_theta={'PER':0,'ORG':-0.2,'LOC':0,'MISC':1}
    select_om_small_total = 1000
    #om_small_theta={'PERSON':0.8,'EVENT':4,'FAC':4,'LANGUAGE':4,'LAW':4,'ORG':1,'PRODUCT':4,'WORK_OF_ART':4,'NORP':1,'LOC':4,'GPE':1}
    
    raw_data_path = f'{dataset_name}/data/train.json'
    
    data_statistic = stas_data_distribution(raw_data_path)
    label_type = get_labels(dataset_name,prefix=False)
    
    # decide the data should selected for each label
    select_data_count = {f'{label}' : 0 for label in label_type}
    total_count = 0
    for item in data_statistic:
        total_count += item[1]
    
    for item in data_statistic:
        label = item[0]
        select_data_count[label]=max(int((item[1]/total_count)*select_om_small_total),1)

    
    om_small_entity = {f'{label}':set() for label in label_type}
    sort_small_entity = {f'{label}':list() for label in label_type}
    
    for split_ind in range(split_number):
        oc_file_name = f'{dataset_name}/PVI/{model_type}_{split_ind}_OC_test.csv'
        om_file_name = f'{dataset_name}/PVI/{model_type}_{split_ind}_OM_test.csv'
        oc_df = pd.read_csv(oc_file_name,quotechar='"')
        om_df = pd.read_csv(om_file_name,quotechar='"')
        om_df.index.name = "index"
        oc_df.index.name = "index"
        df = oc_df.merge(om_df,on='index',how='left',suffixes=('_oc','_om'))


        for index,row in df.iterrows():
        
            cur_label = row["label_om"]
            om_pvi = float(row["PVI_om"])
        
            if om_pvi < om_small_theta[cur_label]:
                if row["text_om"] not in om_small_entity[cur_label]:
                    sort_small_entity[cur_label].append(Entity(row["text_om"],om_pvi))
                    om_small_entity[cur_label].add(row["text_om"])
        
    for label in label_type:
        sort_small_entity[label].sort(key=lambda x:x.pvi)
        
    temp_dict={}
    with open(f'{dataset_name}/data/OM_smaller_total_entity.json','w',encoding='utf-8') as f:
        
        for label,entity_set in sort_small_entity.items():
            temp_dict[label]=[]
            for element in list(entity_set):

                temp_dict[label].append({"om":element['om_text'],"pvi":element['pvi']})
        json.dump(temp_dict,f)

     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default="conll2003")
    parser.add_argument("--split_number",type=int, default= 1)
    parser.add_argument("--tokenizer_path",type=str,default="bert-base-cased")
    parser.add_argument("--model_type",type=str,default="bert")
    parser.add_argument('--label_wise',default=False)
    
    args = parser.parse_args()
    

        
    model_type = args.model_type
    PVI_classification(file_dir = f"{args.dataset_name}/PVI")
    duplicate_removal(dataset_name=args.dataset_name,label_wise=False)
    get_OC_OM(args.dataset_name,args.split_number,tokenizer_path=args.tokenizer_path)
    get_om_small_data(args.dataset_name,args.split_number,tokenizer_path=args.tokenizer_path)