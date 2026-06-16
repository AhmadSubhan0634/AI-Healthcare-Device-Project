import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Symptom Triage Chatbot", layout="wide")
st.title("Smart Symptom Triage Chatbot")
st.caption("Select your symptoms and get a predicted disease and triage level.")

# ── Load models & artefacts ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf_triage  = joblib.load("models/rf_triage.pkl")
    rf_disease = joblib.load("models/rf_disease.pkl")
    le_disease = joblib.load("models/le_disease.pkl")
    le_triage  = joblib.load("models/le_triage.pkl")
    symptom_cols = pickle.load(open("models/symptom_cols.pkl", "rb"))
    return rf_triage, rf_disease, le_disease, le_triage, symptom_cols

try:
    rf_triage, rf_disease, le_disease, le_triage, symptom_cols = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Could not load models: {e}\n\nMake sure the `models/` folder is in the same directory as this app.")

# ── Symptom list ──────────────────────────────────────────────────────────────
# Split into common and rare for better UX
common_symptoms = [c for c in symptom_cols if c.startswith("symptom_")]
rare_symptoms   = [c for c in symptom_cols if c.startswith("rare_symptom_")]

def pretty(col_name):
    """Convert column name to readable label."""
    name = col_name.replace("symptom_", "").replace("rare_symptom_", "").replace("_", " ")
    return name.title()

# ── Triage colour ─────────────────────────────────────────────────────────────
TRIAGE_COLOR = {
    "Emergency":  "🔴 Emergency",
    "Urgent":     "🟠 Urgent",
    "Non-urgent": "🟡 Non-urgent",
    "Self-care":  "🟢 Self-care",
}

# ── Sidebar: symptom selector ─────────────────────────────────────────────────
st.sidebar.header("Select Symptoms")

with st.sidebar.expander("Common Symptoms", expanded=True):
    selected_common = st.multiselect(
        "Choose common symptoms:",
        options=common_symptoms,
        format_func=pretty,
        key="common"
    )

with st.sidebar.expander("Rare Symptoms", expanded=False):
    selected_rare = st.multiselect(
        "Choose rare symptoms:",
        options=rare_symptoms,
        format_func=pretty,
        key="rare"
    )

selected_symptoms = selected_common + selected_rare

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Predict", type="primary", use_container_width=True)



# ── Main area ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Symptoms")
    if selected_symptoms:
        for s in selected_symptoms:
            st.write(f"• {pretty(s)}")
    else:
        st.info("No symptoms selected yet. Use the sidebar to pick symptoms.")

with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        if not models_loaded:
            st.error("Models are not loaded. Cannot predict.")
        elif not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Build feature vector
            input_vec = np.zeros(len(symptom_cols), dtype=np.int8)
            for s in selected_symptoms:
                if s in symptom_cols:
                    input_vec[symptom_cols.index(s)] = 1

            X_input = input_vec.reshape(1, -1)

            # Predict
            disease_enc = rf_disease.predict(X_input)[0]
            triage_enc  = rf_triage.predict(X_input)[0]

            disease_name = le_disease.inverse_transform([disease_enc])[0]
            triage_name  = le_triage.inverse_transform([triage_enc])[0]

            # Disease probabilities (top 5)
            disease_probs = rf_disease.predict_proba(X_input)[0]
            top5_idx      = np.argsort(disease_probs)[::-1][:5]
            top5_diseases = le_disease.inverse_transform(top5_idx)
            top5_probs    = disease_probs[top5_idx]

            # Triage probabilities
            triage_probs = rf_triage.predict_proba(X_input)[0]

            # Display
            label = TRIAGE_COLOR.get(triage_name, triage_name)
            st.metric("Triage Level", label)
            st.metric("Predicted Disease", disease_name.replace("_", " "))

            st.markdown("**Top 5 Possible Diseases:**")
            df_top5 = pd.DataFrame({
                "Disease": [d.replace("_", " ") for d in top5_diseases],
                "Confidence": [f"{p*100:.1f}%" for p in top5_probs],
            })
            st.dataframe(df_top5, hide_index=True, use_container_width=True)

            st.markdown("**Triage Confidence:**")
            df_triage = pd.DataFrame({
                "Triage Level": list(le_triage.classes_),
                "Confidence": [f"{p*100:.1f}%" for p in triage_probs],
            })
            st.dataframe(df_triage, hide_index=True, use_container_width=True)

            st.warning("⚠️ This tool is for educational purposes only. Always consult a qualified healthcare professional.")
    else:
        st.info("Click **Predict** in the sidebar after selecting symptoms.")

# ── Disclaimer footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Models used: Random Forest (Disease & Triage) · Dataset: 100,000 synthetic patient records · 174 symptoms · 46 diseases · 4 triage levels")