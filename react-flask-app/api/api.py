from flask import Flask, request, jsonify
import TestSentence

app = Flask(__name__)


@app.route('/result', methods=['POST'])
def api_post():
    text = request.data
    texty = text.decode('utf-8')
    results = TestSentence.output(texty)
    #print(texty)
    return jsonify(text=results)
