from flask import Flask,render_template,request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/',methods = ['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
       Pregnancies = int(request.form['Pregnancies'])
       Glucose = int(request.form['Glucose'])
       BloodPressure = int(request.form['BloodPressure'])
       SkinThickness = int(request.form['SkinThickness'])
       Insulin = int(request.form['Insulin'])
       BMI = float(request.form['BMI'])
       DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
       Age = int(request.form['Age'])

       x = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

       prediction = model.predict(x)
       
       
       if prediction >= 0.5:
           return render_template('index.html',prediction_text = "Oops you have Diabetes")
       else:
           return render_template('index.html',prediction_text = "Great! You don't have diabetes.") 
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
