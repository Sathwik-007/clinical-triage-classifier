import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Clinical Symptom Triage", page_icon="🏥", layout="centered")

st.title("🏥 Clinical Symptom Triage Classifier")
st.markdown("""
Welcome to the AI Triage Intake System. 
Please select all symptoms you are currently experiencing. Our model will analyze your symptom vector and route you to the appropriate medical department.
""")

# Fetch available symptoms from the API
@st.cache_data
def fetch_symptoms():
    try:
        response = requests.get(f"{API_URL}/symptoms")
        response.raise_for_status()
        return response.json()["symptoms"]
    except requests.exceptions.RequestException:
        st.error("Cannot connect to the Triage API. Ensure the backend is running.")
        return []

available_symptoms = fetch_symptoms()

if available_symptoms:
    selected_symptoms = st.multiselect(
        "Select your symptoms (Type to search):",
        options=available_symptoms
    )

    if st.button("Analyze Symptoms 🩺"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner('Analyzing symptom vectors...'):
                try:
                    payload = {"symptoms": selected_symptoms}
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    response.raise_for_status()
                    
                    data = response.json()
                    predictions = data["predictions"]
                    
                    st.divider()
                    st.subheader("📋 Triage Results")
                    
                    # Display predictions with critical warnings
                    for pred in predictions:
                        if pred["is_critical"]:
                            st.error(f"🚨 **CRITICAL WARNING:** High probability of {pred['disease']}. PLEASE SEEK IMMEDIATE EMERGENCY CARE.")
                        
                        st.write(f"**{pred['rank']}. {pred['disease']}** ({pred['probability_percent']}%)")
                        st.progress(int(pred['probability_percent']))
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to get predictions from API: {e}")

    st.caption("Note: This is an AI portfolio project, not an actual medical advice.")