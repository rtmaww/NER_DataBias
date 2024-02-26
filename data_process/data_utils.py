import os
import json

class Entity():
    def __init__(self,om_text,om_pvi):
        self.om_text = om_text
        self.pvi = om_pvi
    def __getitem__(self,item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise Exception('无关键字')


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, sent_id, words, labels,label_mask,argu_mask=None,entity_pos=None,max_len = 128):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.sent_id = sent_id
        self.words = words
        self.labels = labels
        self.label_mask = label_mask
        if argu_mask:
            self.argu_mask = argu_mask
        else:
            self.argu_mask = label_mask
        self.entity_pos = entity_pos
        self.entities = self._get_entities(max_len)
        
    def __getitem__(self,item):
        if item in self.__dict__:
            return self.__dict__[item]

    def _get_entities(self, max_len=128):
        labels = self.labels
        argu_mask = self.argu_mask
        entities = get_entities(labels,argu_mask)

        for ent_type, start, end in entities:
            # cant assure entity position not out of index
            if start <= max_len - 1 and end > max_len - 1:
                labels = labels[:start-1] + ['O'] * (max_len-start)

        return get_entities(labels,argu_mask)
    
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def read_pvi_file(mode="om",dataset="conll2003"):
    with open(f'{dataset}/data/{mode}_pvi.json','r',encoding='utf-8') as f:
        pvi_dict = json.load(f)
    return pvi_dict


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label, l_mask in zip(example.words, example.labels,example.label_mask):
            word_tokens = tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                if l_mask == 0:
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                else:
                    label_ids.extend([pad_token_label_id] * len(word_tokens))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length, print(len(label_ids), max_seq_length)
        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("sent_id: %s", example.sent_id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("original_label:%s",label_list)
        '''
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    return features


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def get_entities(seq,label_mask):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    label_mask=label_mask+[0]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            if label_mask[begin_offset]==0:
                chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_
    #print(sorted(list(set(chunks)), key=lambda x: x[1]))
    return sorted(list(set(chunks)), key=lambda x: x[1])


def stas_data_distribution(file_dir):
    
    label_dict={}
    with open(file_dir,'r',encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            labels = line['label']
            if 'label_mask' in line:
                label_mask = line['label_mask']
            else:
                label_mask = [0 for i in range(len(labels))]
            for j,label in enumerate(labels):
                if label.startswith("B-"): 
                    if label not in label_dict:
                        label_dict[label] = 0
                    if label_mask[j] != 1 :
                        label_dict[label] += 1
    label_dict = sorted(label_dict.items(),key=lambda x:x[1],reverse=True)
    return label_dict


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        entity_pos = line['entity_pos'] if 'entity_pos' in line else None
        
        if 'argu_mask' not in line:
            #print("no argu_mask..........")
            #print("set argu_mask = label_mask!!!")
            argu_mask=line['label_mask']
        else:
            argu_mask = line['argu_mask']
        examples.append(InputExample(sent_id=line['sent_id'],
                                         words=line['text'],
                                         labels=line['label'],
                                         label_mask=line['label_mask'],
                                         argu_mask=argu_mask,
                                         entity_pos=entity_pos
                                         ))
    return examples


def mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_labels(dataset_name,prefix=True):
    label_path = f'{dataset_name}/data/label.txt'
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.read().splitlines()
            print(len(labels), labels)
        if "O" not in labels:
            labels = ["O"] + labels
        
    else:
        
        data_path = f"{dataset_name}/data/train.json"
        try:
            label_set = set()
            import json
            with open(data_path,'r',encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                labels = line["label"]
                for label in labels:
                    label_set.add(label)
            with open(label_path,'w',encoding="utf-8") as f:
                for item in label_set:
                    f.write(item)
                    f.write("\n")
            labels = list(label_set)
        except Exception as e:
            print(e)
    
    if not prefix:
        labels.remove('O')
        label_set = set()
        for item in labels:
            label_set.add(item.split('-')[-1])
        labels = list(label_set)
        
    return labels