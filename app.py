from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)

try:
    model = pickle.load(open('models/random_forest.pkl', 'rb'))
    scaler = pickle.load(open('models/random_forest_scaler.pkl', 'rb'))
    logging.info("Models loaded successfully ✅")
except Exception as e:
    logging.error(f"Error loading models: {e}")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            pregnancies = float(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            blood_pressure = float(request.form['BloodPressure'])
            skin_thickness = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            dpf = float(request.form['DiabetesPedigreeFunction'])
            age = float(request.form['Age'])

            data = scaler.transform([[pregnancies, glucose, blood_pressure,
                                      skin_thickness, insulin, bmi, dpf, age]])
            prediction = model.predict(data)

            result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

            return render_template('prediction.html', result=result)

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            # Show generic error message to users
            return f"An error occurred: {e}", 500

    else:
        return render_template('form.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
