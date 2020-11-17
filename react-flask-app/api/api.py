from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/result', methods=['POST'])
def api_post():
    text = request.data
    texty = text.decode('utf-8')
    print(texty)
    return jsonify(text=texty)
