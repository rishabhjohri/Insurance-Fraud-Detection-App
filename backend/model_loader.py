import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

def load_all_models():
    model = joblib.load("models/rf_model_with_bert.pkl")
    preprocessor = joblib.load("models/structured_preprocessor.pkl")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, preprocessor, bert_model

def predict_fraud(input_dict, model, preprocessor, bert_model):
    input_df = pd.DataFrame([input_dict])
    structured_cols = ["Customer_Age", "Gender", "Marital_Status", "Occupation", "Coverage_Amount", "Premium_Amount", "Claim_History"]
    X_struct = preprocessor.transform(input_df[structured_cols])
    X_text = bert_model.encode([input_dict["Claim_Description"]])
    from scipy.sparse import hstack
    X_combined = hstack([X_struct, X_text])
    return int(model.predict(X_combined)[0])
