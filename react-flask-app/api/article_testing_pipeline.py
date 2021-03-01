import sys
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
sys.path.append('..\\..\\Related_Articles')
sys.path.append('../../Related_Articles')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import string
#import newscraper
#import pymongo
#from pymongo import MongoClient
import nltk.data
from os import listdir 
from os.path import isfile, join
#from deepcopy import copy
from copy import copy
#import RelatedArticles_five_calls #import getarticles

def analyze_sentences(text):
    sentences = sentence_tokenizer.tokenize(text)

    # Run through algorithm 
    results = TestSentence.output(sentences)

    print('ARTICLE RESULTS:', results['article_score'])
    
    return results


# Incorporate web scrape 

# Take list of URLS --> if web scraper working
# Assume single text file with urls separated by newlines
#url_filename = "../../testing/urls.txt" # include directory
#with 

# OR
# Take directory with text files
# Return list of bias scores
results_folder = "../../testing/results/"
directory_name = "../../testing/Article_Txt_Files/"

# Split into multiple sentences here
nltk.download('punkt')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#directory_specs = ["pleft", "pright", "sleft", "sright", "middle"]
directory_specs = ['middle']

print("Start")

spec_results = {} # dictionary, each key will point to a list of score/article pairs
for spec in directory_specs:
    print(spec)
    cur_dir = directory_name + spec 
    spec_results_path = results_folder + spec + ".json"
    full_spec_results_path = results_folder + spec + "_full.json"
    spec_results = []
    full_spec_results = []
    cur_spec_files = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
    for f in cur_spec_files: 
        article_results = {"filename": f, "article_score":0}
        full_article_results = {"filename": f, "results":0}
        # read in the text
        print(f)
        with open(cur_dir + '/' + f, 'r', encoding='utf-8') as file: 
            article_text = file.read().replace('\n', ' ')

        testsentence_results = analyze_sentences(article_text)

        article_results['article_score'] = testsentence_results['article_score']
        full_article_results['results'] = testsentence_results

        spec_results.append(copy(article_results))
        full_spec_results.append(copy(full_article_results))
    # save spec results and full_spec results to results folder
    with open(spec_results_path, 'w') as fp:
        json.dump(spec_results, fp, indent=4)
    with open(full_spec_results_path, 'w') as fp:
        json.dump(full_spec_results, fp, indent=4)
    