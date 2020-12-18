from flask import Flask, request, jsonify
import json
import TestSentence

app = Flask(__name__)


@app.route('/result', methods=['POST'])
def api_post():
    text = request.data
    texty = text.decode('utf-8')
    dictionary = json.loads(texty)
    print(dictionary['myText'])
    var = dictionary['myText'].lower()
    results = TestSentence.output(var)
    return results #jsonify(text=results)

'''
@app.route('/result-webscrape', methods=['POST'])
def api_post():
    text = request.data
    texty = text.decode('utf-8')
    dictionary = json.loads(texty)
    print(dictionary['myText'])
    var = dictionary['myText'].lower()
    # output from web scraper: list of sentence strings
    # ["my name is sadie", "how are you"]
    #for s in sentences:

    results = TestSentence.output(var)
    return results #jsonify(text=results)'''
