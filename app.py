import pandas as pd
import numpy as np
import flask
import sklearn
from sklearn.preprocessing import StandardScaler

from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model

def predict_outcome(loan_info_list):
    # Load model ***
    ml_model = load_model("8Columns_DeepModel8_Trained.h5")
    
    
    pred_data = pd.DataFrame(loan_info_list)
    scaler = StandardScaler().fit(pred_data)
    scaled_data = scaler.transform(pred_data)
    trans_data = np.array(scaled_data).reshape(1,8)
    prediction = ml_model.predict_classes(trans_data)
    return prediction
def predict_perc(loan_info_list):
        ml_model = load_model("8Columns_DeepModel8_Trained.h5")
        pred_data = pd.DataFrame(loan_info_list)
        scaler = StandardScaler().fit(pred_data)
        scaled_data = scaler.transform(pred_data)
        trans_data = np.array(scaled_data).reshape(1,8)
        percs = ml_model.predict(trans_data)
        return percs[0][1]

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    """Render Home Page"""
    return render_template("index.html")

@app.route("/result", methods = ["POST"])
def generate_prediction():
        if request.method == 'POST':
                res_dict = request.form.to_dict()
                res_list = list(res_dict.values())
                res_list_mapped = list(map(float, res_list))

                pred = predict_outcome(res_list_mapped)
                perc_paid = predict_perc(res_list_mapped)
                perc_paid_scaled = np.round(perc_paid*100, 1)

                if perc_paid >= 0.8:
                        grade = "A"
                elif perc_paid < 0.8 and perc_paid >= 0.7:
                        grade = "B"
                elif perc_paid < 0.7 and perc_paid >= 0.6:
                        grade = "C"
                elif perc_paid < 0.6 and perc_paid >=0.5:
                        grade = "D"
                else:
                        grade = "F"

                if pred == 0:
                        return render_template("result.html", result = "Our model predicts that you will default on your loan.", grade = grade, perc_paid = perc_paid_scaled)
                else:
                        return render_template("result.html", result = "Our model predicts that you will pay off your loan!", grade = grade, perc_paid = perc_paid_scaled)

                

@app.route("/bio")
def load_bio():
        return render_template("bios3.html")
  

       

if __name__ == '__main__':
    app.run(debug = True)