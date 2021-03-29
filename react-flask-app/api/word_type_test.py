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

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# pretrained BERT imports

# other user scripts
from myfeatures import FeatureGenerator # might not need this
from models import AddCombine, BertForMultitaskWithFeatures #, BertForMultitask

def read_lexicon(fp):
        # returns word list as a set
        out = set([
            l.strip() for l in open(fp, errors='ignore') 
            if not l.startswith('#') and not l.startswith(';')
            and len(l.strip().split()) == 1
        ])
        return out

#data = request.data
word = "addict" #data.decode('utf-8')
w_type = "NONE"

DATA_DIRECTORY = '../../ML/data/'
LEXICON_DIRECTORY = DATA_DIRECTORY + 'lexicons/'

lexicons = {
        'assertives': read_lexicon(LEXICON_DIRECTORY + 'assertives_hooper1975.txt'),
        'entailed_arg': read_lexicon(LEXICON_DIRECTORY + 'entailed_arg_berant2012.txt'),
        'entailed': read_lexicon(LEXICON_DIRECTORY + 'entailed_berant2012.txt'), 
        'entailing_arg': read_lexicon(LEXICON_DIRECTORY + 'entailing_arg_berant2012.txt'), 
        'entailing': read_lexicon(LEXICON_DIRECTORY + 'entailing_berant2012.txt'), 
        'factives': read_lexicon(LEXICON_DIRECTORY + 'factives_hooper1975.txt'),
        'hedges': read_lexicon(LEXICON_DIRECTORY + 'hedges_hyland2005.txt'),
        'implicatives': read_lexicon(LEXICON_DIRECTORY + 'implicatives_karttunen1971.txt'),
        'negatives': read_lexicon(LEXICON_DIRECTORY + 'negative_liu2005.txt'),
        'positives': read_lexicon(LEXICON_DIRECTORY + 'positive_liu2005.txt'),
        'npov': read_lexicon(LEXICON_DIRECTORY + 'npov_lexicon.txt'),
        'reports': read_lexicon(LEXICON_DIRECTORY + 'report_verbs.txt'),
        'strong_subjectives': read_lexicon(LEXICON_DIRECTORY + 'strong_subjectives_riloff2003.txt'),
        'weak_subjectives': read_lexicon(LEXICON_DIRECTORY + 'weak_subjectives_riloff2003.txt')
    }

word_tags = []
for l in list(lexicons.keys()):
    #print(lexicons[l], type(lexicons[l]))
    # each lexicon is a set
    if word in lexicons[l]:
        word_tags.append(l)

print(word_tags)

