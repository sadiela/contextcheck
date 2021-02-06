import sys
#sys.path.append('c:\python38\lib\site-packages')
#sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import string
import pymongo
from pymongo import MongoClient


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
    sentences = var.split('.')
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

    '''
    results = {
        sentence_results: [
            {
                words : donald 0.06030 trump 0.06710 sucks 0.12249 ,
                average:  0.08330,
                max_biased_word: sucks 0.12249
            },
            {
                words : I 0.06030 like 0.01710 donuts 0.02249 ,
                average:  0.0033,
                max_biased_word: I 0.06030
            }
        ],
        runtime: 8.332
    }
    '''

'''
@app.route('/result', methods=['POST'])
def api_post():
    text = request.data
    texty = text.decode('utf-8')
    dictionary = json.loads(texty)
    print(dictionary['myText'])
    var = dictionary['myText'].lower()
    results = TestSentence.output(var)
    return results #jsonify(text=results)'''

