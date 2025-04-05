# 🛡️ Insurance Fraud Detection App

## Overview
This app predicts whether a given insurance claim is fraudulent and provides a natural language summary using LLM.

## Tech Stack
- Frontend: Streamlit
- Backend: FastAPI
- ML: RandomForest + BERT + T5

## How to Run

### Backend (FastAPI)
cd backend pip install -r requirements.txt uvicorn main:app --reload

### Frontend (Streamlit)
cd frontend streamlit run app.py

### Folder Structure
- \models/\: Trained ML models
- \'utils/\': Summarization logic
- \'backend/\': FastAPI prediction service
- \'frontend/\': Streamlit frontend
