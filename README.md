# Backend service for SMS Spam Detector app

## What is this app for ?
SMS Spam Detector is an application created for CS-152 Programming Paradigm by Professor Saptarshi Sengupta. This application will serve as a text message validator - users will be able to provide the app a suspecting text message and this app help valid it.

This application is replied on the 2 Machine Learning pre-trained models _RoBERTa_ [HuggingFace](https://huggingface.co/mariagrandury/roberta-base-finetuned-sms-spam-detection) and _BERT Tiny_ [HuggingFace](https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection) to process the language (en-US).

##  The backend service
### How to use ?

-- Clone this repo to your computer
```
git clone https://github.com/chuongtruong/spam-detector-backend/
```
-- Go to the project directory
```
cd ./path-to-project-folder/spam-detector-backend/
``` 
-- Make sure you have Python installed on your machine
```
python --verion
```
-- Create a virtual environment
```
python3 -m venv .venv
```
-- Activate the env and run the backend
```
source .venv/bin/activate (for MAC)
OR
.venv/Scripts/activate (for Windows)  
export FLASK_APP=application.py
export FLASK_ENV=development
python app.py
```

## Link to the Frontend
[spam-detector](https://github.com/chuongtruong/spam-detector)
