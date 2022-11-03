"""
The code is loosely based on this article: 
https://towardsdatascience.com/deploy-an-nlp-pipeline-flask-heroku-bert-f13a302efd9d
"""

from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05-new")

model = AutoModelForSeq2SeqLM.from_pretrained("QuickRead/pegasus-reddit-7e05-new")


def preprocess(inp):
    input_ids = tokenizer(inp, return_tensors="pt").input_ids
    return input_ids
def predict(input_ids):
    outputs = model.generate(input_ids=input_ids)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return res

# Set up flask
app = Flask(__name__)

# Set up flask routes:
# Get:
# Home page

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        inp = request.form['content']
        inp_ids = preprocess(inp)
        print('preprocess result is: \n', inp_ids)
        summary = predict(inp_ids)
        print('Summary is: \n', summary)
        return render_template('index.html', summary=summary)
    print("GETTING get")
    return render_template('index.html', summary="Nothing to summarize")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    