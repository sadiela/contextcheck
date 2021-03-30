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
import newscraper
#import pymongo
#from pymongo import MongoClient
import nltk.data
import RelatedArticles_five_calls #import getarticles
from monkeylearn import MonkeyLearn
import keyword_detection
import tips
from pathlib import Path


app = Flask(__name__)


keyword_api = "d28b596641e1690c696909f66408b6d0ad53e5ca"

'''m_client = MongoClient("mongodb://3.134.119.225")
db = m_client.sentence_results
collection = db.res'''

def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            counter += 1
        else:  # No match found
            if suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def analyze_sentences(text, start_time):
    # Split into multiple sentences here
    #nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)

    # Run through algorithm 
    results = TestSentence.output(sentences)
    
    # Insert data to database # change this to match test article analysis
    '''db_entry = {}
    db_entry['words'] = words
    db_entry['bias_vals'] = bias_values
    db_entry['most_biased'] = most_biased_words
    print("Adding results to DB:")
    collection.insert_one(db_entry)
    print("INSERTED TO DB!")'''

    #print('SENTENCE RESULTS!', results['sentence_results'])
    results['runtime'] = str(time.time() - start_time) + " seconds\n"
    #return_res = results['sentence_results']

    # REMERGE TOKENIZED WORDS (BID ##EN = BIDEN)
    # Make sure sentence parsing is working?!?!?
    return results

@app.route('/result', methods=['POST'])
def api_post():
    print("GET RESULTS!")
    start_time = time.time() # to keep track of analysis runtime

    # get text and format it
    text = request.data
    #print("raw text:", text)
    texty = text.decode('utf-8')
    #texty.translate(str.maketrans('', '', string.punctuation))
    dictionary = json.loads(texty) # why are we loading it into a dictionary and then back out? 
    #print(dictionary['myText'])
    var = dictionary['myText'] #.lower()

    res = analyze_sentences(var, start_time)

    data_path = "./results_jsons/plaintext"
    outdir = get_free_filename('plaintext_res', data_path, suffix='.json')
    with open(outdir, 'w') as fp:
        json.dump(res, fp)

    return res

@app.route('/scrape', methods=['POST'])
def scrape_article():
    start_time = time.time() # to keep track of analysis runtime
    url = request.data
    url = url.decode('utf-8')
    url = json.loads(url)
    print(url)
    url = url['input_url']
    print("PARSING URL")
    res = newscraper.article_parse(url)
    if type(res) is not dict: 
        print("NOT A DICTIONARY:", type(res))
        res = json.loads(res) 

    # res.title, res.author, res.feedText, res.date, res.meta (?)
    print("DONE SCRAPING")
    results = analyze_sentences(res['feedText'], start_time)
    res['bias_results'] = results
    print("DATE:", res['date'])

    ####
    # function gets keywords from title, looks for frequent words in article itself??
    separate_words = res['title'].split(' ')
    ####
    #print(separate_words) # remove 'the', 'a', etc in the future
    
    keywords = keyword_detection.get_keywords(res['title'] + " " + res['feedText'])
    print(keywords)
    related_articles = RelatedArticles_five_calls.getarticles(" ".join(keywords))
    # Call function # return dictionary of {"left":url1, "left-leaning":url2 etc.}
    res['related'] = related_articles

    data_path = "./results_jsons/urls"
    outdir = get_free_filename('article_res', data_path, suffix='.json')
    with open(outdir, 'w') as fp:
        json.dump(res, fp)

    return res

@app.route('/loaderwords', methods=['GET'])
def get_word():
    return tips.get_tips()