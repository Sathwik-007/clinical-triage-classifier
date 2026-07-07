import streamlit as st
import requests
import json

# --- 1. THE FRONTEND DICTIONARY (CACHED) ---
@st.cache_data
def load_symptom_mapping():
    # This print statement will only appear in your terminal ONCE, 
    # proving that Streamlit is safely caching the data!
    print("💾 Loading dictionary into Streamlit cache...") 
    
    with open("../data/processed/ui_symptom_mapping.json", "r") as f:
        return json.load(f)

# Call the function. Streamlit will pull from cache if it already exists.
SYMPTOM_MAPPING = load_symptom_mapping()

# --- 2. FASTAPI CONNECTION ---
# This must match the port where Uvicorn is running
API_URL = "http://127.0.0.1:8000/predict"

# --- 3. UI CONFIGURATION ---
st.set_page_config(
    page_title="DDXPlus Clinical Triage",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Clinical Triage AI")
st.markdown("Select patient symptoms below to generate an AI-powered differential diagnosis using the M4 PyTorch engine.")
st.caption("Note: This is an AI portfolio project, not an actual medical advice.")
st.divider()

# --- 4. THE INTERFACE ---
# Streamlit handles the complex UI state automatically
selected_english_symptoms = st.multiselect(
    "What symptoms is the patient experiencing?",
    options=list(SYMPTOM_MAPPING.keys()),
    placeholder="Choose symptoms..."
)

# --- 5. THE EXECUTION ---
if st.button("Run Diagnostic Engine", type="primary"):
    
    if not selected_english_symptoms:
        st.warning("Please select at least one symptom before running the diagnostic.")
    else:
        # STEP A: Translate English to DDXPlus E_codes
        payload_codes = [SYMPTOM_MAPPING[symp] for symp in selected_english_symptoms]
        
        # STEP B: Send the clean network request to FastAPI
        with st.spinner("Querying the PyTorch Model..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"symptoms": payload_codes}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("differential_diagnosis", [])
                    
                    st.success("Diagnostic Complete")
                    st.subheader("Differential Diagnosis")
                    
                    # STEP C: Render the results beautifully
                    for idx, result in enumerate(results):
                        disease = result["disease"]
                        prob_str = result["probability"]
                        
                        # Convert "18.59%" string back to a float for the progress bar
                        prob_float = float(prob_str.strip('%')) / 100.0
                        
                        st.markdown(f"**{idx + 1}. {disease}** ({prob_str})")
                        st.progress(prob_float)
                        
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Critical Error: Could not connect to the API. Make sure your FastAPI server is running on port 8000!")

