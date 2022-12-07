"""
The code is loosely based on this article: 
https://towardsdatascience.com/deploy-an-nlp-pipeline-flask-heroku-bert-f13a302efd9d
"""

import re
import torch
from flask import Flask, request, jsonify #,render_template
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from langdetect import detect

#tokenizer = AutoTokenizer.from_pretrained("QuickRead/pegasus-reddit-7e05-new")
#model = AutoModelForSeq2SeqLM.from_pretrained("QuickRead/pegasus-reddit-7e05-new")

def remove_emoji(inp):
    """ Function for removing emojis in input """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', inp)

def remove_urls(inp):
    """ Function for removing urls string in input """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', inp)

def remove_html(inp):
    """ Function for removing html string in input """
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', inp)

# return a boolean
def is_english(inp):
    """ Function for filtering out only English input """
    return detect(inp) == 'en'

def preprocess(inp, tokenizer):
    """ Main reprocessing factory """
    # Remove emoji, URL, HTML tag
    clean_input = remove_emoji(inp)
    clean_input = remove_urls(clean_input)
    clean_input = remove_html(clean_input)

    # Filter to get alpha, number, and english only
    if not is_english(clean_input) or len(clean_input) > 1500:
        return torch.zeros(size=(0,1))

    input_ids = tokenizer(clean_input, return_tensors="pt", max_length = 512,
                                        padding = 'max_length',)
    return input_ids

def predict(input_ids, model):
    """ Run model inference """
    # outputs = model.generate(input_ids=input_ids)
    # res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # return res
    with torch.no_grad():
        logits = model(**input_ids).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

# Set up flask
app = Flask(__name__)
CORS(app)

# Setup ML models
tokenizer1 = AutoTokenizer.from_pretrained(
    "mariagrandury/roberta-base-finetuned-sms-spam-detection")

model1= AutoModelForSequenceClassification.from_pretrained(
    "mariagrandury/roberta-base-finetuned-sms-spam-detection")

tokenizer2 = AutoTokenizer.from_pretrained(
    "mrm8488/bert-tiny-finetuned-sms-spam-detection")

model2 = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/bert-tiny-finetuned-sms-spam-detection")

@app.route('/', methods=['POST', 'GET'])
def index():
    """ Main route """
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json = request.json
            print("Request json: ", json)
        # inp = request.form['content']
        inp = json['input']
        print("Input data from POST reque7ui8st: ", inp)
        # Preprocess for 2 models: 
        inp_ids1 = preprocess(inp, tokenizer1)
        inp_ids2 = preprocess(inp, tokenizer2)
        print(inp_ids1, inp_ids2)
        if len(inp_ids1.input_ids) == 0 or len(inp_ids2.input_ids) == 0:
            # error case
            err_message = {
                "Error":"Input Error: \
                    Make sure that input is English or has max character count of 1000"
            }
            return jsonify(err_message)
        print('Tokenized input for model \
            "mariagrandury/roberta-base-finetuned-sms-spam-detection": \n', inp_ids1)
        print('Tokenized input for model \
            "mrm8488/bert-tiny-finetuned-sms-spam-detection": \n', inp_ids2)

        # Get labels from 2 inputs: 
        label1 = predict(inp_ids1, model1)
        label2 = predict(inp_ids2, model2)

        print('Returned label for model \
            "mariagrandury/roberta-base-finetuned-sms-spam-detection": \n', label1)
        print('Returned label for model \
            "mrm8488/bert-tiny-finetuned-sms-spam-detection": \n', label2)
        ret_obj = {
            'roberta':{
                'label': label1,
                'url': 'https://huggingface.co/mariagrandury/roberta-base-finetuned-sms-spam-detection'
                },
            'bert-tiny':{
                'label':label2,
                'url':'https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection'
                }
        }

        return jsonify(ret_obj)

    # Send a GET request, the return object should just include the tittle
    ret_obj = {
        "Title": "Hello From Spam Detection App"
    }
    print("GETTING get")
    return jsonify(ret_obj)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    