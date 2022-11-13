"""
The code is loosely based on this article: 
https://towardsdatascience.com/deploy-an-nlp-pipeline-flask-heroku-bert-f13a302efd9d
"""

from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
from langdetect import detect
import re

tokenizer = AutoTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05-new")
model = AutoModelForSeq2SeqLM.from_pretrained("QuickRead/pegasus-reddit-7e05-new")

def remove_emoji(input):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', input)

def remove_urls(input):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', input)

def remove_html(input):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', input)

# return a boolean
def is_english(input):
    return detect(input) == 'en'

def preprocess(inp):
    # Remove emoji, URL, HTML tag
    clean_input = remove_emoji(inp)
    clean_input = remove_urls(clean_input)
    clean_input = remove_html(clean_input)

    # Filter to get alpha, number, and english only
    if not is_english(clean_input) or len(clean_input) > 1500:
        return torch.zeros(size=(0,1))

    input_ids = tokenizer(clean_input, return_tensors="pt", max_length = 512,
                                        padding = 'max_length',).input_ids
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
        if inp_ids.nelement() == 0:
            # error case
            return render_template('index.html', summary="Input Error: Make sure that input is English or has max character count of 1000")
        print('preprocess result is: \n', inp_ids)
        summary = predict(inp_ids)
        print('Summary is: \n', summary)
        return render_template('index.html', summary=summary)
    print("GETTING get")
    return render_template('index.html', summary="Nothing to summarize")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    