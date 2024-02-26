from datasets import load_dataset
import os
import json
from transformers import AutoTokenizer



def get_ent_pos(text,tokenizer):
        pos = 1 
        text = text.strip().split()
        for item in text:
            if item == tokenizer.mask_token:
                break
            else:
                ##calculate the count of subword token
                pos += 1
        temp_text = " ".join(text[:pos])
        length = len(tokenizer.tokenize(temp_text))
        return length

def generate_OC_OM(output_path,tokenizer_path,tokenizer_name,train_id):
    
    data_files={}
    data_files['train'] =  os.path.join(output_path,f'train_{train_id}.json')
    data_files['test'] = os.path.join(output_path,f'test_{train_id}.json')
    raw_dataset = load_dataset('json',data_files=data_files)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for key in raw_dataset:
        filter_count = 0
        texts, labels, ents_pos, ents, doc_ids, sent_ids = [], [], [], [], [],[]
        text_list = raw_dataset[key]['text']
        label_list = raw_dataset[key]['label']
        doc_id_list = raw_dataset[key]['doc_id']
        sent_id_list = raw_dataset[key]['sent_id']
        for j,(text, label,doc_id,sent_id) in enumerate(zip(text_list,label_list,doc_id_list,sent_id_list)):


            for i,(word,tag) in enumerate(zip(text,label)):  
                if tag.startswith('B-') :

                    

                    ent_start = i
                    ent_end = i+1
                    while ent_end < len(label) and label[ent_end].startswith('I-'):
                        ent_end += 1
                        
                    new_ent = text[ent_start:ent_end]
                    new_ent = " ".join(new_ent)
                    new_text = text[:ent_start] + [tokenizer.mask_token] + text[ent_end:]
                    new_text = " ".join(new_text)
                    new_label = tag.split('-')[-1]

                    mask_sub_word_pos = get_ent_pos(new_text,tokenizer)
                    max_process_token = 128
                 
                    if mask_sub_word_pos < max_process_token:
                        pos = [mask_sub_word_pos,mask_sub_word_pos+1]
                        

                        ents.append(new_ent)
                        texts.append(new_text)
                        labels.append(new_label)
                        doc_ids.append(doc_id)
                        sent_ids.append(sent_id)
                        ents_pos.append(pos)
                        '''
                        if j < 10:
                            print(text)
                            print(new_text)
                            print(label)
                            print(new_label)
                            print(pos)
                        '''
                    else:
                        # max process token for tokenizer
                        filter_count += 1
               
        print(f"filter data count:{filter_count}")
       
        model_name = tokenizer_name.split('-')[0]
        
        oc_file_name = key + f'_{model_name}_{train_id}_OC.json'
        path = os.path.join(output_path,oc_file_name)
        with open(path,'w',encoding ='utf-8') as f:
            for doc_id,sent_id,text,label,ent_pos in zip(doc_ids,sent_ids,texts,labels,ents_pos):
                json.dump({'doc_id':doc_id,'sent_id':sent_id,'text':text,'label':label,'pos':[ent_pos[0],ent_pos[0]+1]},f,ensure_ascii=False)
                f.write('\n')
                
        om_file_name = key+f'_{model_name}_{train_id}_OM.json'
        path = os.path.join(output_path,om_file_name)
        with open(path,'w',encoding ='utf-8') as f:
            for doc_id,sent_id,ent,label in zip(doc_ids,sent_ids,ents,labels):
                json.dump({'doc_id':doc_id,'sent_id':sent_id,'text':ent,'label':label},f,ensure_ascii=False)
                f.write('\n')
                
        null_file_name = key+f'_{model_name}_{train_id}_null.json'
        path = os.path.join(output_path,null_file_name)
        with open(path,'w',encoding ='utf-8') as f:
            for doc_id,sent_id,label in zip(doc_ids,sent_ids,labels):
                json.dump({'doc_id':doc_id,'sent_id':sent_id,'text':" ",'label':label},f,ensure_ascii=False)
                f.write('\n')
