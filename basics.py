from flask import Flask, render_template, request
import pickle
model = pickle.load(open('modelv2.pkl', 'rb'))
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = request.form["age"]
    hypertension = request.form["hypertension"]
    bmi = request.form["bmi"]
    HbA1c_level = request.form["HbA1c_level"]
    glucose = request.form["glucose"]
    dd = np.array([[
        float(age), 
        float(hypertension), 
        float(bmi), 
        float(HbA1c_level),
        float(glucose)
        ]])
    prediction = model.predict(dd)
    
    print(prediction)
    # prediction = model.predict(dd)
    return render_template("result.html" , data=prediction)

if __name__ == "__main__":
    app.run(debug=True)