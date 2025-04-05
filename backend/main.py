from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_all_models, predict_fraud
from utils.summarizer import generate_summary, make_prompt

app = FastAPI()
model, preprocessor, bert_model = load_all_models()

class Claim(BaseModel):
    Claim_Description: str
    Customer_Age: float
    Gender: str
    Marital_Status: str
    Occupation: str
    Coverage_Amount: int
    Premium_Amount: int
    Claim_History: float

@app.post("/predict")
def predict(claim: Claim):
    row = claim.dict()
    is_fraud = predict_fraud(row, model, preprocessor, bert_model)
    summary = generate_summary(make_prompt(row))
    return {"fraud": bool(is_fraud), "summary": summary}
