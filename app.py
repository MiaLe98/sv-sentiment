import flask
from flask import Flask,render_template,url_for,request
import pandas as pd 
import joblib
import nltk
import numpy as np

nltk.download('punkt')

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():

    classifier = joblib.load("model-73acc.cls")
    
    if request.method == 'POST':
        def format_sentence(sent):
            return np.array([sent])

        message = request.form['message']
        my_prediction = classifier.predict(format_sentence(message))


    return render_template('result.html',prediction = my_prediction)

@app.route('/opinion',methods=['GET', 'POST'])
def opinion(): 
    if request.method == 'POST': 
        req = request.form['options'] 
        print(req)
        return render_template('home.html')

    return render_template('opinion.html', opinion = req)
    
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 80, debug=True)