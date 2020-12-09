############################
#######  Imports  ##########
############################
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

import features

###################
# DIRECTORY PATHS #
###################
## UPDATE THESE!!!
DATA_DIRECTORY = 'data/'
LEXICON_DIRECTORY = DATA_DIRECTORY + 'lexicons/'
PRYZANT_DATA = DATA_DIRECTORY + 'bias_data/WNC/'
#IMPORTS = 
training_data = PRYZANT_DATA + 'biased.word.train'
testing_data = PRYZANT_DATA + 'biased.word.test'
categories_file = PRYZANT_DATA + 'revision_topics.csv'
pickle_directory = 'pickle_dir/'
cache_dir = DATA_DIRECTORY + 'cache/'
model_save_dir = 'trained_models/'


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

##################################################
#### FUNCTIONS FOR DATA FORMATTING/PROCESSING ####
##################################################

# not sure what this function does
def get_tok_labels(s_diff):
  pre_tok_labels = []
  post_tok_labels = []
  for tag, chunk in s_diff:
    if tag == '=':
      pre_tok_labels += [0] * len(chunk)
      post_tok_labels += [0] * len(chunk)
    elif tag == '-':
      pre_tok_labels += [1] * len(chunk) # 1 in pre if word deleted in post
    elif tag == '+':
      post_tok_labels += [1] * len(chunk) # 1 in post if word added in post
    else: 
      pass
  return pre_tok_labels, post_tok_labels 
  # returns returns list of 0s, list of 1s for both pre and post edit sentences

  ### LEAVING OUT NOISE_SEQ FUNCTION FOR NOW ###

# PADDING TO MAX_SEQ_LENGTH
def pad(id_arr, pad_idx):
  max_seq_len = 80
  return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

def get_examples(data_path, tok2id, max_seq_len, 
                 noise=False, add_del_tok=False,
                 categories_path=None):

    skipped = 0 
    out = defaultdict(list)

    with open(data_path, 'r+', encoding='utf-8') as data_file: 
        lines = [line for line in data_file.readlines()]

    #print(lines[0])
    #input()

    for i, (line) in enumerate(tqdm(lines)):
        parts = line.strip().split('\t')

        #print(parts)
        #input()

        # if there pos/rel info
        if len(parts) == 7:
            [revid, pre, post, _, _, pos, rels] = parts
        # no pos/rel info
        elif len(parts) == 5:
            [revid, pre, post, _, _] = parts
            pos = ' '.join(['<UNK>'] * len(pre.strip().split()))
            rels = ' '.join(['<UNK>'] * len(pre.strip().split()))
        # broken line
        else:
            skipped += 1
            continue

        # break up tokens
        tokens = pre.strip().split()
        post_tokens = post.strip().split()
        rels = rels.strip().split()
        pos = pos.strip().split()

        # get diff + binary diff masks
        tok_diff = diff(tokens, post_tokens)
        pre_tok_labels, post_tok_labels = get_tok_labels(tok_diff)
                   
        # make sure everything lines up    
        if len(tokens) != len(pre_tok_labels) \
            or len(tokens) != len(rels) \
            or len(tokens) != len(pos) \
            or len(post_tokens) != len(post_tok_labels):
            skipped += 1
            continue

        # leave room in the post for start/stop and possible category/class token
        if len(tokens) > max_seq_len - 1 or len(post_tokens) > max_seq_len - 1:
            skipped += 1
            continue

        # category info if provided
        # TODO -- if provided but not in diyi's data, we fill with random...is that ok?

        categories = np.random.uniform(size=43)   # 43 = number of categories
        categories = categories / sum(categories) # normalize


        # add start + end symbols to post in/out
        post_input_tokens = ['行'] + post_tokens
        post_output_tokens = post_tokens + ['止'] 

        # shuffle + convert to ids + pad
        try:
            pre_toks = tokens

            pre_ids = pad([tok2id[x] for x in pre_toks], 0)
            post_in_ids = pad([tok2id[x] for x in post_input_tokens], 0)
            post_out_ids = pad([tok2id[x] for x in post_output_tokens], 0)
            pre_tok_label_ids = pad(pre_tok_labels, EDIT_TYPE2ID['mask'])
            post_tok_label_ids = pad(post_tok_labels, EDIT_TYPE2ID['mask'])
            rel_ids = pad([REL2ID.get(x, REL2ID['<UNK>']) for x in rels], 0)
            pos_ids = pad([POS2ID.get(x, POS2ID['<UNK>']) for x in pos], 0)
        except KeyError:
            # TODO FUCK THIS ENCODING BUG!!!
            skipped += 1
            continue

        input_mask = pad([0] * len(tokens), 1)
        pre_len = len(tokens)

        out['pre_ids'].append(pre_ids)
        out['pre_masks'].append(input_mask)
        out['pre_lens'].append(pre_len)
        out['post_in_ids'].append(post_in_ids)
        out['post_out_ids'].append(post_out_ids)
        out['pre_tok_label_ids'].append(pre_tok_label_ids)
        out['post_tok_label_ids'].append(post_tok_label_ids)
        out['rel_ids'].append(rel_ids)
        out['pos_ids'].append(pos_ids)
        out['categories'].append(categories)

    print('SKIPPED ', skipped)
    return out


########################
## DEFINE DATA LOADER ##
########################
# get data loader for data in data_path
def get_dataloader(data_path, tok2id, batch_size, 
                   pickle_path=None, test=False, noise=False, add_del_tok=False, 
                   categories_path=None, sort_batch=True):
    #global ARGS

    def collate(data):
        if sort_batch:
            # sort by length for packing/padding
            data.sort(key=lambda x: x[2], reverse=True)
        # group by datatype
        [
            src_id, src_mask, src_len, 
            post_in_id, post_out_id, 
            pre_tok_label, post_tok_label,
            rel_ids, pos_ids, categories
        ] = [torch.stack(x) for x in zip(*data)]

        # cut off at max len of this batch for unpacking/repadding
        max_len = max(src_len)
        data = [
            src_id[:, :max_len], src_mask[:, :max_len], src_len, 
            post_in_id[:, :max_len+10], post_out_id[:, :max_len+10],    # +10 for wiggle room
            pre_tok_label[:, :max_len], post_tok_label[:, :max_len+10], # +10 for post_toks_labels too (it's just gonna be matched up with post ids)
            rel_ids[:, :max_len], pos_ids[:, :max_len], categories
        ]

        return data

    if pickle_path is not None and os.path.exists(pickle_path):
        print("pickle file exists!")
        examples = pickle.load(open(pickle_path, 'rb'))
    else:
        examples = get_examples(
            data_path=data_path, 
            tok2id=tok2id,
            max_seq_len=80, #ARGS.max_seq_len,
            noise=False, #noise,
            add_del_tok=False, #add_del_tok,
            categories_path=None)#categories_path)

        pickle.dump(examples, open(pickle_path, 'wb'))

    data = TensorDataset(
        torch.tensor(examples['pre_ids'], dtype=torch.long),
        torch.tensor(examples['pre_masks'], dtype=torch.uint8), # byte for masked_fill()
        torch.tensor(examples['pre_lens'], dtype=torch.long),
        torch.tensor(examples['post_in_ids'], dtype=torch.long),
        torch.tensor(examples['post_out_ids'], dtype=torch.long),
        torch.tensor(examples['pre_tok_label_ids'], dtype=torch.float),  # for compartin to enrichment stuff
        torch.tensor(examples['post_tok_label_ids'], dtype=torch.float),  # for loss multiplying
        torch.tensor(examples['rel_ids'], dtype=torch.long),
        torch.tensor(examples['pos_ids'], dtype=torch.long),
        torch.tensor(examples['categories'], dtype=torch.float))


    dataloader = DataLoader(
        data,
        sampler=(SequentialSampler(data) if test else RandomSampler(data)),
        collate_fn=collate,
        batch_size=batch_size)

    return dataloader, len(examples['pre_ids'])


CUDA = (torch.cuda.device_count() > 0)

# GET DATA LOADERS!
train_dataloader, num_train_examples = get_dataloader(
    data_path=training_data,
    tok2id=tok2id,
    batch_size=32,
    pickle_path=pickle_directory + 'train_data4.p',
    categories_path=None #categories_file
  )

eval_dataloader, num_eval_examples = get_dataloader(
    data_path=testing_data,
    tok2id=tok2id,
    batch_size=32,
    pickle_path=pickle_directory + 'test_data4.p',
    categories_path=None #categories_file
  )

print(num_train_examples, num_eval_examples)

# DEFINE MODEL

# BERT initialization params
config = 'bert-base-uncased'
cls_num_labels = 43
tok_num_labels = 3
tok2id = tok2id


#####################
# CLASS DEFINITIONS #
#####################

class AddCombine(nn.Module):
    def __init__(self, hidden_dim, feat_dim, layers, dropout_prob, small=False,
            out_dim=-1, pre_enrich=False, include_categories=False,
            category_emb=False, add_category_emb=False):
        super(AddCombine, self).__init__()

        self.include_categories = include_categories
        if include_categories:
            feat_dim += 43

        if layers == 1:
            self.expand = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.Dropout(dropout_prob))
        else:
            waist_size = min(feat_dim, hidden_dim) if small else max(feat_dim, hidden_dim)
            self.expand = nn.Sequential(
                nn.Linear(feat_dim, waist_size),
                nn.Dropout(dropout_prob),
                nn.Linear(waist_size, hidden_dim),
                nn.Dropout(dropout_prob))
        
        if out_dim > 0:
            self.out = nn.Linear(hidden_dim, out_dim)
        else:
            self.out = None

        if pre_enrich:
            self.enricher = nn.Linear(feature_size, feature_size)        
        else:
            self.enricher = None

        # manually set cuda because module doesn't see these combiners for bottom         
        if CUDA:
            self.expand = self.expand.cuda()
            if out_dim > 0:
                self.out = self.out.cuda()
            if self.enricher is not None:
                self.enricher = self.enricher.cuda()

    def forward(self, hidden, feat, categories=None):
        if self.include_categories:
            categories = categories.unsqueeze(1)
            categories = categories.repeat(1, features.shape[1], 1)
            if self.add_category_emb:
                features = features + categories
            else:
                features = torch.cat((features, categories), -1)

        if self.enricher is not None:
            feat = self.enricher(feat)
    
        combined = self.expand(feat) + hidden
    
        if self.out is not None:
            return self.out(combined)

        return combined

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
        global ARGS
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        cls_logits = self.cls_classifier(pooled_output)
        cls_logits = self.cls_dropout(cls_logits)

        # NOTE -- dropout is after proj, which is non-standard
        tok_logits = self.tok_classifier(sequence_output)
        tok_logits = self.tok_dropout(tok_logits)

        return cls_logits, tok_logits

class BertForMultitaskWithFeatures(PreTrainedBertModel): 
    
    def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None, lexicon_feature_bits=1):
        super(BertForMultitaskWithFeatures, self).__init__(config)

        self.bert = BertModel(config)

        self.featureGenerator = features.FeatureGenerator(POS2ID, REL2ID, tok2id=tok2id, pad_id=0, lexicon_feature_bits=lexicon_feature_bits)
        nfeats = 90 if lexicon_feature_bits == 1 else 118; 

        # hidden_size = 512
        # nfeats
        # combiner_layers = 1
        # hidden_dropout_prob = 
        # small_waist = false
        # out dim
        # pre_enrich = false
        # include_categories = False
        # category_emb = 
        # add_category_emb = 
        self.tok_classifier = AddCombine(config.hidden_size, nfeats, 1,
                config.hidden_dropout_prob, False,
                out_dim=tok_num_labels, pre_enrich=False,
                include_categories=False,
                category_emb=False,
                add_category_emb=False)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                rel_ids=None, pos_ids=None, categories=None, pre_len=None)
        
        features = self.featurizer.featurize_batch(
            input_ids.detach().cpu().numpy(), 
            rel_ids.detach().cpu().numpy(), 
            pos_ids.detach().cpu().numpy(), 
            padded_len=input_ids.shape[1])
        features = torch.tensor(features, dtype=torch.float)

        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.cls_dropout(pooled_output)
        cls_logits = self.cls_classifier(pooled_output)

        if ARGS.category_emb:
            categories = self.category_embeddings(
                categories.max(-1)[1].type(
                    'torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

        tok_logits = self.tok_classifier(sequence_output, features, categories)

        return cls_logits, tok_logits


lexicon_feature_bits = 1

# define model!!
'''model = BertForMultitask.from_pretrained(
    'bert-base-uncased',
    cls_num_labels=cls_num_labels,
    tok_num_labels=tok_num_labels, 
    cache_dir=cache_dir,
    tok2id=tok2id)'''

model = BertForMultitaskWithFeatures.from_pretrained(
    'bert-based-uncased',
    cls_num_labels=cls_num_labels,
    tok_num_labels=tok_num_labels,
    tok2id=None, 
    lexicon_feature_bits=1)


def build_optimizer(model, num_train_steps, learning_rate):
#global ARGS

    '''if ARGS.tagger_from_debiaser:
        parameters = list(model.cls_classifier.parameters()) + list(
            model.tok_classifier.parameters())
        parameters = list(filter(lambda p: p.requires_grad, parameters))
        return optim.Adam(parameters, lr=ARGS.learning_rate)
    else:'''
    param_optimizer = list(model.named_parameters())
    param_optimizer = list(filter(lambda name_param: name_param[1].requires_grad, param_optimizer))
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    return BertAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            warmup=0.1,
                            t_total=num_train_steps)


def build_loss_fn(debias_weight=None):
    global ARGS
    
    if debias_weight is None:
        debias_weight = 1 # default #ARGS.debias_weight
    
    weight_mask = torch.ones(3) #ARGS.num_tok_labels)
    weight_mask[-1] = 0

    if CUDA:
        weight_mask = weight_mask.cuda()
        criterion = CrossEntropyLoss(weight=weight_mask).cuda()
        per_tok_criterion = CrossEntropyLoss(weight=weight_mask, reduction='none').cuda()
    else:
        criterion = CrossEntropyLoss(weight=weight_mask)
        per_tok_criterion = CrossEntropyLoss(weight=weight_mask, reduction='none')


    def cross_entropy_loss(logits, labels, apply_mask=None):
        return criterion(
            logits.contiguous().view(-1, 3), #ARGS.num_tok_labels), 
            labels.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

    def weighted_cross_entropy_loss(logits, labels, apply_mask=None):
        # weight mask = where to apply weight (post_tok_label_id from the batch)
        weights = apply_mask.contiguous().view(-1)
        weights = ((debias_weight - 1) * weights) + 1.0

        per_tok_losses = per_tok_criterion(
            logits.contiguous().view(-1, 3), # ARGS.num_tok_labels), 
            labels.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))
        per_tok_losses = per_tok_losses * weights

        loss = torch.mean(per_tok_losses[torch.nonzero(per_tok_losses)].squeeze())

        return loss

    if debias_weight == 1.0:
        loss_fn = cross_entropy_loss
    else:
        loss_fn = weighted_cross_entropy_loss

    return loss_fn

def to_probs(logits, lens):
    per_tok_probs = softmax(np.array(logits)[:, :, :2], axis=2)
    pos_scores = per_tok_probs[:, :, -1]
    
    out = []
    for score_seq, l in zip(pos_scores, lens):
        out.append(score_seq[:l].tolist())
    return out

def run_inference(model, eval_dataloader, loss_fn, tokenizer):
    #global ARGS

    out = {
        'input_toks': [],
        'post_toks': [],

        'tok_loss': [],
        'tok_logits': [],
        'tok_probs': [],
        'tok_labels': [],

        'labeling_hits': []
    }

    for step, batch in enumerate(tqdm(eval_dataloader)):
        #if False and step > 2:
        #    continue
        if step%5 == 0:
          if CUDA:
              batch = tuple(x.cuda() for x in batch)

          ( 
              pre_id, pre_mask, pre_len, 
              post_in_id, post_out_id, 
              tok_label_id, _,
              rel_ids, pos_ids, categories
          ) = batch

          with torch.no_grad():
              _, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
                  rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                  pre_len=pre_len)
              tok_loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
          out['input_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in pre_id.cpu().numpy()]
          out['post_toks'] += [tokenizer.convert_ids_to_tokens(seq) for seq in post_in_id.cpu().numpy()]
          out['tok_loss'].append(float(tok_loss.cpu().numpy()))
          logits = tok_logits.detach().cpu().numpy()
          labels = tok_label_id.cpu().numpy()
          out['tok_logits'] += logits.tolist()
          out['tok_labels'] += labels.tolist()
          out['tok_probs'] += to_probs(logits, pre_len)
          out['labeling_hits'] += tag_hits(logits, labels)

    return out

def train_for_epoch(model, train_dataloader, loss_fn, optimizer):
    global ARGS
    
    losses = []
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        #if ARGS.debug_skip and step > 2:
        #    continue
    
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        ( 
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch
        _, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
            rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
            pre_len=pre_len)
        loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())

    return losses

def is_ranking_hit(probs, labels, top=1):
    global ARGS
    
    # get rid of padding idx
    [probs, labels] = list(zip(*[(p, l)  for p, l in zip(probs, labels) if l != 3 - 1 ]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0

def tag_hits(logits, tok_labels, top=1):
    #global ARGS
    
    probs = softmax(np.array(logits)[:, :, : 3 - 1], axis=2)

    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top) 
        for prob_dist, tok_label in zip(probs, tok_labels)
    ]
    return hits


epochs = 4
train_batch_size = 32 
learning_rate = 3e-5
optimizer = build_optimizer(
    model, int((num_train_examples * epochs) / train_batch_size),
    learning_rate)
loss_fn = build_loss_fn()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print('INITIAL EVAL...')
model.eval()
results = run_inference(model, eval_dataloader, loss_fn, tokenizer)

# TRAIN MODEL!!
# run_inference
# train_for_epoch

print(results['input_toks'][8])
print(results['tok_labels'][8])
#print(results['tok_logits'][8])

summary = "" #SummaryWriter(model_save_dir)

#input()
#writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
#writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), 0)
summary += 'eval/tok_loss' + str(np.mean(results['tok_loss'])) + '\n'
summary += 'eval/tok_acc' + str(np.mean(results['labeling_hits'])) + '\n'

print('TRAINING...')
model.train()
for epoch in range(epochs):
    print('STARTING EPOCH ', epoch)
    losses = train_for_epoch(model, train_dataloader, loss_fn, optimizer)
    #writer.add_scalar('train/loss', np.mean(losses), epoch + 1)
    summary += 'train/loss' + str(np.mean(losses)) + '\n'

        # eval
    print('EVAL...')
    model.eval()
    results = run_inference(model, eval_dataloader, loss_fn, tokenizer)
    #writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), epoch + 1)
    #writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), epoch + 1)
    summary += 'eval/tok_loss' + str(np.mean(results['tok_loss'])) + '\n'
    summary += 'eval/tok_acc' + str(np.mean(results['labeling_hits'])) + '\n'


    model.train()

    print('SAVING...')
    torch.save(model.state_dict(), model_save_dir + 'model_%d.ckpt' % epoch)

print("Writing summary to file")
filename = model_save_dir + 'training_output.txt'
text_file = open(filename, "wt")
n = text_file.write(summary)
text_file.close()