import os
import copy
import logging
import random
import json


from data_process.data_utils import InputExample,read_pvi_file

logger = logging.getLogger(__name__)


def get_rand_ent_idxes(examples,aug_rate=-1, aug_om_small_mode="None",dataset_name="", max_seq_len=128):
    oc_bigger_theta={'ORG':1.6,'PER':1.6,'LOC':1.55,'MISC':2.4}
    oc_pvi = read_pvi_file(mode="oc",dataset=dataset_name)
    cont_ent_idxes = []
    change_examples=[]
    change_inds=[]
    
    aug_num = int(len(examples)*aug_rate)
    cur_aug_num = 0

    switch_skip = 0
    oc_skip =0
    if aug_om_small_mode=="None":
        switch_theta=1.2
    else:
        switch_theta = 0.8
        
    for example_ind,example in enumerate(examples):
        switch_num = random.random()
        if cur_aug_num < aug_num and switch_num < switch_theta:
            argu_mask = example.argu_mask
                
            ent_span_idxes=[]
            for entity_ind,entity in enumerate(example.entities):
                ent_type,pos_start,pos_end = entity
                if aug_om_small_mode == "select_oc_bigger":
                    
                    ent_context = " ".join(example.words[:pos_start]+['<split>']+example.words[pos_end+1:])
                    try:
                        ent_context_pvi = oc_pvi[ent_type][ent_context]["pvi"]
                    except:
                        continue
                    if pos_end < max_seq_len and ent_context_pvi > oc_bigger_theta[ent_type]:
                        ent_span_idxes .append(entity_ind) 
                    
                else :
                    ent_span_idxes .append(entity_ind) 

            
            if len(ent_span_idxes)>0:
                cur_aug_num += 1
                select_aug_index = random.choice(ent_span_idxes)
                if cur_aug_num<6:
                    print(f"argu_mask:{argu_mask}")
                    print(f'select_aug_index:{select_aug_index}')
                    print(f"entities:{example.entities[select_aug_index]}")
                cont_ent_idxes.append(select_aug_index)
                change_examples.append(example)
                change_inds.append(example_ind)
            else:   # no entity instance 
                if len(example.entities) >0:
                    oc_skip +=1     
                cont_ent_idxes.append(-1)
                #print("no entity to do augmentation")
        else:
            if switch_num > switch_theta:
                switch_skip +=1
            cont_ent_idxes.append(-1)
            
    print("dataset sample total:{}".format(len(examples) if aug_num==-1 else aug_num))
    print(f"available aug sample total:{cur_aug_num}")
    print(f"switch skip the examples num:{switch_skip}")
    print(f"oc small skip the examples num:{oc_skip}")
    
    return cont_ent_idxes,change_examples,change_inds


def build_aug_examples(
        examples,
        aug_mode,
        aug_rate=-1,
        aug_om_small_mode="None",
        dataset_name="", 
        max_seq_len=128,
        om_small_json=None,
        change_type = "aug",
):
    
    cont_ent_idxes,raw_examples,raw_inds = get_rand_ent_idxes(examples,aug_rate=aug_rate, aug_om_small_mode=aug_om_small_mode,dataset_name=dataset_name, max_seq_len=128)
    
    with open(om_small_json,'r',encoding='utf-8') as f:
        os.path.exists(om_small_json)
        om_small_json = json.load(f)
    
    
    
    change_examples = []
    total_om_small = 0
    for i, example in enumerate(examples):
      
        if om_small_json and aug_mode == "om_small":
            cont_example = copy.deepcopy(example)
            if len(example.entities) >0 and cont_ent_idxes[i]!=-1 :
                total_om_small +=1
                entities = cont_example.entities
                label, start, end = entities[cont_ent_idxes[i]]
                
                rep_words = om_small_ent_rep([label, start, end],om_small_json)
                
                cont_words = cont_example.words[:start] + rep_words + cont_example.words[end+1:]
                cont_labels = cont_example.labels[:start] + \
                                ["B-" + label] + ["I-" + label] * (len(rep_words) -1) + \
                                cont_example.labels[end + 1:]
                label_mask=cont_example.label_mask[:start]+[0]*len(rep_words)+cont_example.label_mask[end+1:]
                argu_mask = cont_example.argu_mask[:start]+[0]*len(rep_words)+cont_example.argu_mask[end+1:]
                assert(len(label_mask)==len(cont_labels))
                assert(len(cont_labels)==len(cont_words))
                cont_example = InputExample(sent_id=example.sent_id, words=cont_words, labels=cont_labels,
                                            label_mask=label_mask,argu_mask=argu_mask)
            
                
       
        change_examples.append(cont_example)
   
    print(f'aug_om_small:{total_om_small}')
    
    if change_type == "aug":
        change_examples.extend(raw_examples)
        
    for i in range(5):
        print(f"words:{examples[raw_inds[i]].words} \n labels:{examples[raw_inds[i]].labels}")
        print(f"words:{change_examples[raw_inds[i]].words} \n labels:{change_examples[raw_inds[i]].labels}")
        

    return change_examples


def om_small_ent_rep(entity, auxiliary_json=None):
   
    label = entity[0]
    
    length = len(auxiliary_json[label])
    select_end=length-1
    replace_entity_ind = random.randint(0,select_end)
    new_entity_words = auxiliary_json[label][replace_entity_ind]['om'].split(' ')
   

   
    return new_entity_words
