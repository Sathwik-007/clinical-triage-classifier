from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle as pkl
import os
import numpy as np

app = FastAPI(title="Clinical Triage Inference API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../data/processed/clinical_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pkl.load(f)

# extracting model components
W = model['model_weights']
b = model['model_bias']
diseases = model['diseases']
symptom_weight = model['symptom_weight'] 
symptoms_list = model['symptoms']

# Create a user-friendly version of the symptoms for the UI
ui_symptoms = [s.replace('_', ' ').title() for s in symptoms_list]
HIGH_RISK = ["Heart attack", "Paralysis (brain hemorrhage)", "Stroke", "Heart failure"]

class TriageRequest(BaseModel):
    symptoms: list[str]

def softmax(Z):
    """Computes the softmax of a vector."""
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(Z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# API Endpoints
@app.get('/symptoms')
def get_symptom_list():
    """Returns the list of symptoms available in the model."""
    return {'symptoms': ui_symptoms} 

@app.post('/predict')
def predict_disease(request: TriageRequest):
    """Predicts the disease based on the provided symptoms."""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")

    X_input = np.zeros((1, len(symptoms_list)))

    for ui_symptom in request.symptoms:
        # Convert the UI string ("Chest Pain") back to the dataset string ("chest_pain")
        original_symp = ui_symptom.lower().replace(' ', '_')
        
        if original_symp in symptoms_list:
            idx = symptoms_list.index(original_symp)
            weight = symptom_weight.get(original_symp, 1) # Now this will successfully find the weight!
            X_input[0, idx] = weight

    # Calculate the logits and apply softmax
    Z = np.dot(X_input, W) + b
    probabilities = softmax(Z)[0]

    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    results = []
    for i, idx in enumerate(top_3_indices):
        disease_name = str(diseases[idx])
        prob = float(probabilities[idx])
        
        is_critical = bool(i == 0 and disease_name in HIGH_RISK and prob > 0.30)
        
        results.append({
            "rank": i + 1,
            "disease": disease_name,
            "probability_percent": round(prob * 100, 1),
            "is_critical": is_critical
        })
        
    return {"predictions": results}