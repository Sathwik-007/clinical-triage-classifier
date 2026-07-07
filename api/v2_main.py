from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import json
import os

# --- 1. PYTORCH ARCHITECTURE (Must match training exactly) ---
class ClinicalTriageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClinicalTriageModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- 2. API SETUP & GLOBAL VARIABLES ---
app = FastAPI(title="DDXPlus Clinical Triage API")

# We will load these into RAM when the server starts
model = None
mappings = {}
english_to_e_code = {}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 3. THE DATA MODEL ---
class PatientRequest(BaseModel):
    symptoms: list[str]  # e.g., ["chest pain", "fever"]

# --- 4. SERVER STARTUP (Loading the Brain) ---
@app.on_event("startup")
async def load_model_and_dicts():
    global model, mappings, english_to_e_code
    print(f"🚀 Booting API Engine on: {device}")

    # 1. Load the PyTorch Dictionaries
    mappings_path = "../v2_scale_ddxplus/ddxplus_mappings.pkl"
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    num_symptoms = len(mappings["all_symptoms"])
    num_classes = len(mappings["all_diseases"])

    # 2. Load the PyTorch Model
    model_path = "../v2_scale_ddxplus/ddxplus_model.pth"
    model = ClinicalTriageModel(num_symptoms, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Freeze weights for inference!

    # 3. Build the English -> E_Code Dictionary
    # We parse the DDXPlus json to map human words to Kaggle codes
    evidences_path = "../data/raw/release_evidences.json"
    with open(evidences_path, "r") as f:
        evidences_data = json.load(f)
        
    for e_code, details in evidences_data.items():
        # Grab the English name and make it lowercase for easy matching
        english_name = details.get("name", "").lower()
        english_to_e_code[english_name] = e_code
        
    print("✅ System Ready. Waiting for patients...")

# --- 5. THE PREDICTION ENDPOINT ---
@app.post("/predict")
async def triage_patient(request: PatientRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model engine offline.")

    # STEP A: English -> E_Code
    patient_e_codes = []
    for symptom in request.symptoms:
        clean_symptom = symptom.lower().strip()
        if clean_symptom in english_to_e_code:
            patient_e_codes.append(english_to_e_code[clean_symptom])
        else:
            print(f"⚠️ Warning: Symptom '{clean_symptom}' not found in medical dictionary.")

    # STEP B: E_Code -> Tensor Index
    x_tensor = torch.zeros(len(mappings["all_symptoms"]), dtype=torch.float32)
    symptom_to_idx = mappings["symptom_to_idx"]
    
    for e_code in patient_e_codes:
        if e_code in symptom_to_idx:
            x_tensor[symptom_to_idx[e_code]] = 1.0

    # Move tensor to GPU and add a "batch" dimension of size 1 (Shape: [1, 131])
    x_tensor = x_tensor.unsqueeze(0).to(device)

    # STEP C: Matrix Calculus (The Forward Pass)
    with torch.no_grad():
        logits = model(x_tensor)
        # Apply Softmax to get real 0.0 to 1.0 percentages
        probabilities = torch.softmax(logits, dim=1)[0] 

    # STEP D: Tensor Index -> English Disease Name
    disease_to_idx = mappings["disease_to_idx"]
    idx_to_disease = {v: k for k, v in disease_to_idx.items()} # Reverse the dictionary
    
    # Get the top 3 most likely diseases
    top_3_probs, top_3_indices = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        disease_name = idx_to_disease[top_3_indices[i].item()]
        probability = top_3_probs[i].item() * 100
        results.append({
            "disease": disease_name,
            "probability": f"{probability:.2f}%"
        })

    return {
        "provided_symptoms": request.symptoms,
        "recognized_codes": patient_e_codes,
        "differential_diagnosis": results
    }