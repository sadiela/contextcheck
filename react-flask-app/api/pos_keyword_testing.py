#################
#### Imports ####
#################

# C:\Users\sadie\AppData\Roaming\Python\Python38\Scripts\pipenv shell

import sys
import time
import os

sys.path.append('../../ML')
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
sys.path.append('..\..\ML')

import numpy as np
import statistics
import spacy
import newscraper
import TestSentence
import json
from monkeylearn import MonkeyLearn

import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk import tokenize
from operator import itemgetter
import math

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

import newscraper

# pretrained BERT imports
import pytorch_pretrained_bert.modeling as modeling
from pytorch_pretrained_bert.modeling import BertModel, BertSelfAttention, BertPreTrainedModel, BertForTokenClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

# other user scripts
from myfeatures import FeatureGenerator # might not need this
from models import AddCombine, BertForMultitaskWithFeatures #, BertForMultitask

import keyword_detection

nlp = spacy.load("en_core_web_sm")

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

config = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(config, os.getcwd() + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

nltk.download('stopwords')
# read an article 
url = "https://www.foxnews.com/politics/gop-senators-immigration-deal-border-crisis"
res = newscraper.article_parse(url)
#print(res)
res_obj = json.loads(res) 
#print(res_obj)
# res.title, res.author, res.feedText, res.date, res.meta (?)
print("DONE SCRAPING")

text = res_obj['feedText']

keywords = keyword_detection.get_keywords(text)
print(keywords)

'''
print(title)
ml = MonkeyLearn("d28b596641e1690c696909f66408b6d0ad53e5ca")
model_id = "ex_YCya9nrn"
result = ml.extractors.extract(model_id, title)
#print(result.body)
for item in result.body:
  print(item['extractions'])

#nltk.download('punkt')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sentence_tokenizer.tokenize(text)
results = TestSentence.output(sentences)
print(results)

for s in sentences: 
    sentence_dat = nlp(s)
    sentence_pos = [i.pos_ for i in sentence_dat]
    sentence_tokens = [i.text.lower() for i in sentence_dat]
    final_tokens = []
    final_pos = [] 
    for word, pos in zip(sentence_tokens, sentence_pos):
        cur_tok = tokenizer.tokenize(word)
        for c in cur_tok:
            final_tokens.append(c)
            final_pos.append(pos)
    print(len(final_pos), len(final_tokens))
    print(final_pos, final_tokens)
    input("Continue...")
'''

    