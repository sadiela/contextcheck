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
