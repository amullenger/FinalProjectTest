import pandas as pd
import numpy as np
import flask

from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model

def predict_outcome(loan_info_list):
    # Load model ***
    ml_model = load_model("8Columns_DeepModel8_Trained.h5")
    pred_data = np.array(loan_info_list).reshape(1, 8)
    prediction = ml_model.predict_classes(pred_data)
    return prediction[0]


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

                if pred == 0:
                        return render_template("result.html", result = "Our model predicts that you will default on your loan!")
                else:
                        return render_template("result.html", result = "Our model predicts that you will pay off your loan.")

                return render_template("result.html", result = pred)

@app.route("/bio")
def load_bio():
        return render_template("bios3.html")
  

       

if __name__ == '__main__':
    app.run(debug = True)