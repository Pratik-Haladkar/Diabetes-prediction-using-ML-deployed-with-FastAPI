from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# create FastAPI app (THIS IS IMPORTANT)
app = FastAPI()

# load trained model
model = pickle.load(open("diabetes_model.sav", "rb"))

# input schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/diabetes_prediction")
def predict_diabetes(data: DiabetesInput):

    input_data = np.array([[ 
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    prediction = model.predict(input_data)

    return {
        "prediction": "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    }

