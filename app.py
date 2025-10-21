from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler 
import pickle
import pandas as pd



app = Flask(__name__)
model = pickle.load(open('models/model (1).pkl','rb'))
scaler = pickle.load(open('models/scaler (2).pkl','rb'))
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


       data = scaler.transform([[pregnancies,glucose,blood_pressure,skin_thickness ,insulin, bmi, dpf, age]])
       predict = model.predict(data)
       if predict[0]==1:
           result='Diabetic'
       else:
           result='Non-Diabetic'    
           

       return render_template('prediction.html',result =result )

    except Exception as e:
            # Show error on the page (only in dev mode)
            return f"Error: {e}", 500

           

 else:
    return render_template('form.html')
  

if __name__ == "__main__" :
    app.run(host="0.0.0.0")
    
     
