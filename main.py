from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class Email(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Spam Detector API is running"}

@app.post("/predict")
def predict(email: Email):
    transformed = vectorizer.transform([email.text])
    prediction = model.predict(transformed)[0]
    
    result = "spam" if prediction == 1 else "not spam"
    return {"email": email.text, "prediction": result}