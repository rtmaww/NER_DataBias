from os import truncate
from transformers import AutoModel,BertPreTrainedModel,RobertaPreTrainedModel,BertModel,RobertaModel
import torch
from torch import nn
class EntityClassificationOutput():
    def __init__(self,loss,logits):
        self.loss = loss
        self.logits = logits

class BertEntityClassification(BertPreTrainedModel):
    def __init__(
        self,
        config
        ):
        super(BertEntityClassification,self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel.from_pretrained("bert-base-cased")
        
        self.dropout = nn.Dropout( config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,self.num_labels)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None,ent_pos=None):
        output = self.bert(input_ids,attention_mask,token_type_ids)
       
        sequence_output = output[0]
       
        if ent_pos != None:
            classifier_input = list()
            for idx in range(sequence_output.shape[0]):
                mask_ind_pos = ent_pos[idx]
                if mask_ind_pos[1] > 128:
                    print("entity over")
                    mask_ind_pos[1] = 128
                entity_representation = torch.zeros(sequence_output.shape[-1]).cuda()
                for mask_pos in range(mask_ind_pos[0],mask_ind_pos[1]):
                    entity_representation += sequence_output[idx][mask_pos][:]
                ent_token_num = mask_ind_pos[1]-mask_ind_pos[0]
                entity_representation /= ent_token_num
                classifier_input.append(entity_representation)

                
            classifier_input = torch.stack(classifier_input)
            classifier_input.cuda()
        else:
            classifier_input = output[1]

       
        classifier_input = self.dropout(classifier_input)
        logits = self.classifier(classifier_input)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits,labels)
        return EntityClassificationOutput(loss,logits)

     

class RobertaEntityClassification(RobertaPreTrainedModel):
    def __init__(
        self,
        config,
        model_name="roberta-base"
        ):
        super(RobertaEntityClassification,self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,self.num_labels)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None,ent_pos=None):
      
        output = self.encoder(input_ids,attention_mask,token_type_ids)
      
        sequence_output = output[0]
      
        if ent_pos != None:
            classifier_input = list()
            for idx in range(sequence_output.shape[0]):
                mask_ind_pos = ent_pos[idx]
                if mask_ind_pos[1] > 128:
                    print("entity over")
                    mask_ind_pos[1] = 128
                entity_representation = torch.zeros(sequence_output.shape[-1]).cuda()
                for mask_pos in range(mask_ind_pos[0],mask_ind_pos[1]):
                    entity_representation += sequence_output[idx][mask_pos][:]
                ent_token_num = mask_ind_pos[1]-mask_ind_pos[0]
                entity_representation /= ent_token_num
                classifier_input.append(entity_representation)

               
                
            classifier_input = torch.stack(classifier_input)
            classifier_input.cuda()
        else:
            classifier_input = output[1]

       
        classifier_input = self.dropout(classifier_input)
        logits = self.classifier(classifier_input)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits,labels)
        return EntityClassificationOutput(loss,logits)



