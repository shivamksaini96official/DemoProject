import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## import elastic net model and standard scaler
elastic_model = joblib.load(open('models/final_model.pkl','rb'))
standard_scaler = joblib.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
  if request.method == 'POST':
    TV = float(request.form.get('TV'))
    Radio = float(request.form.get('Radio'))
    Newspaper = float(request.form.get('Newspaper'))

    new_scaled_data = standard_scaler.transform([[TV, Radio, Newspaper]])
    result = elastic_model.predict(new_scaled_data)

    return render_template('home.html', results=np.round(result[0],2))
  else:
    return render_template('home.html')


if __name__ == "__main__":
  app.run(host="0.0.0.0")