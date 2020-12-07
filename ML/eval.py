# Imports
#from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertSelfAttention
import pytorch_pretrained_bert.modeling as modeling
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tqdm import tqdm
import sys

import pickle
import os
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
import argparse
import sklearn.metrics as metrics
#from simplediff import diff
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertModel, BertSelfAttention
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

#####################
### DIF #############
#####################
def softmax(x, axis=None):
  x=x-x.max(axis=axis, keepdims=True)
  y= np.exp(x)
  return y/y.sum(axis=axis, keepdims=True)

def diff(old, new):

    # Create a map from old values to their indices
    old_index_map = dict()
    for i, val in enumerate(old):
        old_index_map.setdefault(val,list()).append(i)

    # Find the largest substring common to old and new.
    # We use a dynamic programming approach here.
    # 
    # We iterate over each value in the `new` list, calling the
    # index `inew`. At each iteration, `overlap[i]` is the
    # length of the largest suffix of `old[:i]` equal to a suffix
    # of `new[:inew]` (or unset when `old[i]` != `new[inew]`).
    #
    # At each stage of iteration, the new `overlap` (called
    # `_overlap` until the original `overlap` is no longer needed)
    # is built from the old one.
    #
    # If the length of overlap exceeds the largest substring
    # seen so far (`sub_length`), we update the largest substring
    # to the overlapping strings.

    overlap = dict()
    # `sub_start_old` is the index of the beginning of the largest overlapping
    # substring in the old list. `sub_start_new` is the index of the beginning
    # of the same substring in the new list. `sub_length` is the length that
    # overlaps in both.
    # These track the largest overlapping substring seen so far, so naturally
    # we start with a 0-length substring.
    sub_start_old = 0
    sub_start_new = 0
    sub_length = 0

    for inew, val in enumerate(new):
        _overlap = dict()
        for iold in old_index_map.get(val,list()):
            # now we are considering all values of iold such that
            # `old[iold] == new[inew]`.
            _overlap[iold] = (iold and overlap.get(iold - 1, 0)) + 1
            if(_overlap[iold] > sub_length):
                # this is the largest substring seen so far, so store its
                # indices
                sub_length = _overlap[iold]
                sub_start_old = iold - sub_length + 1
                sub_start_new = inew - sub_length + 1
        overlap = _overlap

    if sub_length == 0:
        # If no common substring is found, we return an insert and delete...
        return (old and [('-', old)] or []) + (new and [('+', new)] or [])
    else:
        # ...otherwise, the common substring is unchanged and we recursively
        # diff the text before and after that substring
        return diff(old[ : sub_start_old], new[ : sub_start_new]) + \
               [('=', new[sub_start_new : sub_start_new + sub_length])] + \
               diff(old[sub_start_old + sub_length : ],
                       new[sub_start_new + sub_length : ])

##########################################################
#### SET UP ID DICTIONARIES FOR WORDS, RELATIONS, POS ####
##########################################################
## UPDATE THESE!!!
DATA_DIRECTORY = '/usr4/ec523/sadiela/ec463_proj/data/'
LEXICON_DIRECTORY = DATA_DIRECTORY + 'lexicons/'
PRYZANT_DATA = DATA_DIRECTORY + 'bias_data/WNC/'
#IMPORTS = 
training_data = PRYZANT_DATA + 'biased.word.train'
testing_data = PRYZANT_DATA + 'biased.word.test'
categories_file = PRYZANT_DATA + 'revision_topics.csv'
pickle_directory = '/usr4/ec523/sadiela/ec463_proj/results/pickle_dir/'
cache_dir = DATA_DIRECTORY + 'cache/'
model_save_dir = '/usr4/ec523/sadiela/ec463_proj/results/saved_models/'


RELATIONS = [
  'det', # determiner (the, a)
  'amod', # adjectival modifier
  'nsubj', # nominal subject
  'prep', # prepositional modifier
  'pobj', # object of preposition
  'ROOT', # root
  'attr', # attribute
  'punct', # punctuation
  'advmod', # adverbial modifier
  'compound', # compound
  'acl', # clausal modifier of noun (adjectivial clause)
  'agent', # agent
  'aux', # auxiliary
  'ccomp', # clausal complement
  'dobj', # direct object
  'cc', # coordinating conjunction 
  'conj', # conjunct
  'appos', # appositional 
  'nsubjpass', # nsubjpass
  'auxpass', # auxiliary (passive)
  'poss', # poss
  'nummod', # numeric modifier
  'nmod', # nominal modifier
  'relcl', # relative clause modifier
  'mark', # marker
  'advcl', # adverbial clause modifier
  'pcomp', # complement of preposition
  'npadvmod', # noun phrase as adverbial modifier
  'preconj', # pre-correlative conjunction
  'neg', # negation modifier
  'xcomp', # open clausal complement
  'csubj', # clausal subject
  'prt', # particle
  'parataxis', # parataxis
  'expl', # expletive
  'case', # case marking
  'acomp', # adjectival complement
  'predet', # ??? 
  'quantmod', # modifier of quantifier
  'dep', # unspecified dependency
  'oprd', # object predicate
  'intj', # interjection
  'dative', # dative
  'meta', # meta modifier
  'csubjpass', # clausal subject (passive)
  '<UNK>' # unknown
]

REL2ID = {x: i for i,x in enumerate(RELATIONS)}

# PARTS OF SPEECH
POS_TAGS = [
  'DET', # determiner (a, an, the)
  'ADJ', # adjective (big, old, green, first)
  'NOUN', # noun (girl, cat, tree)
  'ADP', # adposition (in, to, during)
  'NUM', # numeral (1, 2017, one, IV)
  'VERB', # verb (runs, running, eat, ate)
  'PUNCT', # punctuation (., (, ), ?)
  'ADV', # adverb (very, tomorrow, down)
  'PART', # particle ('s, not)
  'CCONJ', # coordinating conjunction (and, or, but)
  'PRON', # pronoun(I, you, he, she)
  'X', # other (fhefkoskjsdods)
  'INTJ', # interjection (hello, psst, ouch, bravo)
  'PROPN', # proper noun (Mary, John, London, HBO) 
  'SYM', # symbol ($, %, +, -, =)
  '<UNK>' # unknown
]

POS2ID = {x: i for i, x in enumerate(POS_TAGS)}

EDIT_TYPE2ID = {'0':0, '1':1, 'mask':2}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', os.getcwd() + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

# PADDING TO MAX_SEQ_LENGTH
def pad(id_arr, pad_idx):
  max_seq_len = 60
  return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))


class BertForMultitask(BertPreTrainedModel):

    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitask, self).__init__(config)
        self.bert = BertModel(config)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
        
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, rel_ids=None, pos_ids=None, categories=None, pre_len=None):
        #global ARGS
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        cls_logits = self.cls_classifier(pooled_output)
        cls_logits = self.cls_dropout(cls_logits)

        # NOTE -- dropout is after proj, which is non-standard
        tok_logits = self.tok_classifier(sequence_output)
        tok_logits = self.tok_dropout(tok_logits)

        return cls_logits, tok_logits

def to_probs(logits, lens):
    per_tok_probs = softmax(np.array(logits)[:, :, :2], axis=2)
    pos_scores = per_tok_probs[-1, :, :]
    #print(per_tok_probs)
    #print("Pos scores:", pos_scores, len(pos_scores))
    #print(lens)
    out = []
    #for score_seq, l in zip(pos_scores, lens):
    out.append(pos_scores[:].tolist())

    return out

# Take one sentence ... 
def run_inference(model, ids, tokenizer):
    #global ARGS
    # we will pass in one sentence, no post_toks, 

    out = {
        'input_toks': [],
        #'tok_loss': [],
        'tok_logits': [],
        'tok_probs': []
        #'tok_labels': [],
        #'labeling_hits': []
        #'input_len': 0
    }

    #for step, batch in enumerate(tqdm(eval_dataloader)):
        #if False and step > 2:
        #    continue
    #if CUDA:
    #    batch = tuple(x.cuda() for x in batch)
    pre_len = len(ids)

    with torch.no_grad():
        _, tok_logits = model(ids, attention_mask=None,
            rel_ids=None, pos_ids=None, categories=None,
            pre_len=None) # maybe pre_len
        #tok_loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
    out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in ids.cpu().numpy()]
    #out['post_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in post_in_id.cpu().numpy()]
    #out['tok_loss'].append(float(tok_loss.cpu().numpy()))
    logits = tok_logits.detach().cpu().numpy()
    #labels = tok_label_id.cpu().numpy()
    out['tok_logits'] += logits.tolist()
    #out['tok_labels'] += labels.tolist()
    out['tok_probs'] += to_probs(logits, pre_len)
    #out['input_len'] = pre_len
    #out['labeling_hits'] += tag_hits(logits, labels)

    return out

def test_sentence(s): 
    POS2ID = {x: i for i, x in enumerate(POS_TAGS)}

    EDIT_TYPE2ID = {'0':0, '1':1, 'mask':2}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', os.getcwd() + '/cache')
    tok2id = tokenizer.vocab
    tok2id['<del>'] = len(tok2id)

    # BERT initialization params
    config = 'bert-base-uncased'
    cls_num_labels = 43
    tok_num_labels = 3
    tok2id = tok2id

    # define model!!
    model = BertForMultitask.from_pretrained(
        'bert-base-uncased',
        cls_num_labels=cls_num_labels,
        tok_num_labels=tok_num_labels, 
        cache_dir=cache_dir,
        tok2id=tok2id)
_
    # Load model 
    saved_model_path = model_save_dir + 'model_3.ckpt'
    model.load_state_dict(torch.load(saved_model_path))

    tokens = s.strip().split()
    length = len(tokens)
    ids = pad([tok2id[x] for x in tokens], 0)
    #print(ids)
    ids = torch.LongTensor(ids)
    #ids = ids.type(torch.LongTensor)
    #print(ids, ids.size())
    ids = ids.unsqueeze(0)
    #print(ids, ids.size(1))

    output = run_inference(model, ids, tokenizer)
    return output, length 

out, length = test_sentence(sentence) 
print("Results:")

#print(length)
#print(out['input_toks'][0][:29])
#print(out['tok_probs'][0][:29])

avg_sum = 0

for l in out['tok_probs'][0][:length]:
    #print(l.index(max(l)))
    avg_sum += l[1]

print("Average bias: ", avg_sum/length)