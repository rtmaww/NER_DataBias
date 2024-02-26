from data_process.data_utils import get_labels,InputExample,read_pvi_file,mk_dir
import json
import random
import argparse


def context_aug(single_reduction_rate,dataset="conll2003",oc_select_mode="random"):
    data_path=f"{dataset}/data/train.json"
    label_type = get_labels(dataset,prefix=False)
    reduction_rate = {label:single_reduction_rate for label in label_type}

    entity_total_list={label:list() for label in label_type}
    context_total_list= {label:list() for label in label_type}
    entity_total_pvi_list={label:list() for label in label_type}
    context_total_pvi_list={label:list() for label in label_type}
    entity_total_set={label:set() for label in label_type}
    context_total_set= {label:set() for label in label_type}
    
    oc_pvi = read_pvi_file("oc",dataset)
    om_pvi = read_pvi_file("om",dataset)
    
    om_pvi_not_in_list = {label:0 for label in label_type}
    oc_pvi_not_in_list = {label:0 for label in label_type}


    with open(data_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        
   

    raw_examples=[]
    for line in lines:
        line = json.loads(line)
        labels = line['label']
        label_mask = line['label_mask']
        argu_mask = label_mask
        example = InputExample(sent_id=line['sent_id'],words=line['text'],labels=labels,label_mask=label_mask,argu_mask=argu_mask)
        entities = example.entities
        for entity in entities:
            entity_type,pos_start,pos_end = entity
            entity_text = example.words[pos_start:pos_end+1]
            join_entity_text = '<split>'.join(entity_text)
            context_text = example.words[0:pos_start]+['<split>']+example.words[pos_end+1:]
            join_context_text = " ".join(context_text)
            context_label = example.labels[0:pos_start]+['<split>']+example.labels[pos_end+1:]
            context_label_mask_pre = [0 if element=='O' else 1 for element in labels[0:pos_start] ]
            context_label_mask_post= [0 if element=='O' else 1 for element in labels[pos_end+1:] ]
            context_label_mask = context_label_mask_pre+[0]+context_label_mask_post
            #join_context_text = " ".join(context_text)
            entity_total_list[entity_type].append(join_entity_text)
            #context_total_list[entity_type].append(" ".join(context_text))

            # if join_entity_text not in entity_total_set[entity_type]:
            entity_total_set[entity_type].add(join_entity_text)
            if join_entity_text in om_pvi[entity_type]:
                entity_total_pvi_list[entity_type].append({"entity":join_entity_text,"pvi":om_pvi[entity_type][join_entity_text]})
            else:
                om_pvi_not_in_list[entity_type] += 1
                
            context_total_list[entity_type].append({"context":context_text,"label_mask":context_label_mask,"pos":pos_start,"label":context_label})
        
            # if join_context_text not in context_total_set[entity_type]:
            context_total_set[entity_type].add(join_context_text)
            if join_context_text in oc_pvi[entity_type]:
                context_total_pvi_list[entity_type].append({"context":join_context_text,"pvi":oc_pvi[entity_type][join_context_text]['pvi']})
            else:
                oc_pvi_not_in_list[entity_type] += 1
                    
        raw_examples.append(example)
    
    entity_candidate_set={label:set() for label in label_type}
    entity_small_candidate_set={label:set() for label in label_type}
    entity_bigger_candidate_set={label:set() for label in label_type}
    entity_candidate_list={label:list() for label in label_type}
    context_candidate_list={label:list() for label in label_type}
    context_candidate_set={label:set() for label in label_type}
   

    for k,v in context_total_pvi_list.items():
        if oc_select_mode == "bigger":
            v.sort(key=lambda x:x['pvi'])
            
        elif oc_select_mode == "small":
            v.sort(key=lambda x:x['pvi'],reverse=True)
        select_num = int(reduction_rate[k]*len(context_total_set[k]))
        #if reduction_rate[k] > 0.5:
            #extend_num={'ORG':700,'PER':700,'LOC':600,'MISC':200}
        print(select_num)
        # bigger 0.1
        # extend_num={'ORG':600,'PER':600,'LOC':400,'MISC':350}
        # bigger 0.1
        #extend_num={'ORG':600,'PER':600,'LOC':400,'MISC':350}
        if oc_select_mode=="bigger" or oc_select_mode=="small":
            extend_num = int(select_num * 0.15)
            print(f"ddl_pvi:{v[select_num-1]['pvi']}")
            print(f"extend_pvi:{v[select_num + extend_num-1]['pvi']}")
            temp_list = random.sample(v[:select_num+extend_num],select_num)
        else:
            print("random_select")
            temp_list = random.sample(v,select_num)

            
        for item in temp_list:
            context_candidate_set[k].add(item['context'])
        for element in context_total_list[k]:
            if " ".join(element["context"]) not in context_candidate_set[k]:
                context_candidate_list[k].append(element)

    context_new_examples=[]
    context_not_change_num={label:0 for label in label_type}
    for example_ind,example in enumerate(raw_examples):
        ori_texts = example.words
        ori_labels = example.labels
        entities = example.entities
        for ent_ind,entity in enumerate(entities):      
            new_texts,new_labels,new_label_mask=[],[],[]     
            entity_type,pos_start,pos_end = entity
            ent_text = ori_texts[pos_start:pos_end+1]
            ori_context = " ".join(ori_texts[0:pos_start]+['<split>']+example.words[pos_end+1:])
            context_obj = random.choice(context_candidate_list[entity_type])
            context = context_obj['context']
            join_context = " ".join(context)
            if ori_context in context_candidate_set[entity_type]:
                ent_pos = context_obj['pos']
                context_label = context_obj['label']
                label_mask = context_obj['label_mask']
                new_texts = context[:ent_pos] + ent_text + context[ent_pos+1:]
                new_labels = context_label[:ent_pos] + ori_labels[pos_start:pos_end+1] + context_label[ent_pos+1:]
                new_label_mask = label_mask[:ent_pos] + [0 for i in range(pos_start,pos_end+1)] + label_mask[ent_pos+1:]
            else:
                context_not_change_num[entity_type] += 1
                new_texts = ori_texts
                new_labels = ori_labels
                label_mask_pre = [0 if element=='O' else 1 for element in ori_labels[0:pos_start] ]
                label_mask_post= [0 if element=='O' else 1 for element in ori_labels[pos_end+1:] ]
                new_label_mask = label_mask_pre+[0 for i in range(pos_start,pos_end+1)]+label_mask_post
            
            context_new_examples.append(InputExample(sent_id=example.sent_id,words=new_texts,labels=new_labels,label_mask=new_label_mask,argu_mask=new_label_mask))
        if len(entities) ==0:
            context_new_examples.append(example)
    
    print(oc_pvi_not_in_list)
    print(om_pvi_not_in_list)
    print(f'context_not_change_num:{context_not_change_num}')
   
    
   
    folder = f'{dataset}/data/replace_context_{oc_select_mode}2bigger'
    mk_dir(folder)
    with open(f'{folder}/replace{single_reduction_rate}_train.json','w',encoding='utf-8') as f:
        for example in context_new_examples:
            json.dump({'sent_id':example.sent_id,'text':example.words,'label':example.labels,'label_mask':example.label_mask},f)
            f.write('\n')

    return context_new_examples
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default="conll2003")
    parser.add_argument("--reduction_rate",type=float,default=0.2)
    parser.add_argument("--oc_select_mode",type=str,default="random")
    #random
    
    args = parser.parse_args()
    

    
    context_aug(single_reduction_rate=args.reduction_rate,dataset=args.dataset_name,oc_select_mode = args.oc_select_mode)
    
    



    
