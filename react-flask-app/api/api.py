import sys
#sys.path.append('c:\python38\lib\site-packages')
#sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import string
import newscraper
import pymongo
from pymongo import MongoClient
import nltk.data


app = Flask(__name__)

'''m_client = MongoClient("mongodb://3.134.119.225")
db = m_client.sentence_results
collection = db.res'''

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
    var = dictionary['myText'].lower()

    # Split into multiple sentences here
    nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(var)
    #sentences = var.split('. ')
    print(sentences)

    # Run through algorithm 
    results = TestSentence.output(sentences[:-1])
    
    #print("Average bias: ", avg_sum/len(words)

    # Insert data to database # change this to match test article analysis
    '''db_entry = {}
    db_entry['words'] = words
    db_entry['bias_vals'] = bias_values
    db_entry['most_biased'] = most_biased_words
    print("Adding results to DB:")
    collection.insert_one(db_entry)
    print("INSERTED TO DB!")'''

    print('SENTENCE RESULTS!', results['sentence_results'])
    results['runtime'] = str(time.time() - start_time) + " seconds\n"
    #return_res = results['sentence_results']
    return results

@app.route('/scrape', methods=['POST'])
def scrape_article():
    url = request.data
    url = url.decode('utf-8')
    url = json.loads(url)
    print(url)
    url = url['input_url']
    res = newscraper.article_parse(url)
    return res

