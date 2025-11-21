import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(page_title="Mental Health - Fixed Loader", layout="wide")
st.title("Mental Health - Prediction (Fixed loader)")

# possible model paths (check models/ first, then root)
POSSIBLE = [
    "models/pipeline_voting.pkl",
    "models/pipeline_rf.pkl",
    "models/pipeline_lr.pkl",
    "pipeline_voting.pkl",
    "pipeline_rf.pkl",
    "pipeline_lr.pkl"
]

found = {}
for p in POSSIBLE:
    if os.path.exists(p):
        found[os.path.basename(p)] = p

if not found:
    st.sidebar.error("Model files not found. Expected one of: " + ", ".join(POSSIBLE))
    st.sidebar.write("Files in repo root:", os.listdir("."))
    st.stop()

st.sidebar.write("Detected model files:")
for k, v in found.items():
    st.sidebar.write(f"- {k}  →  {v}")

# Model selection
default_name = "pipeline_voting.pkl" if "pipeline_voting.pkl" in found else list(found.keys())[0]
model_choice = st.sidebar.selectbox("Model to use", list(found.keys()), index=list(found.keys()).index(default_name))
model_path = found[model_choice]

@st.cache_resource
def load_model_cached(path):
    """Load model with caching to avoid repeated disk IO across reruns."""
    return joblib.load(path)

# Load model on button click and store in session_state so it persists across reruns
if st.sidebar.button("Load model"):
    try:
        model = load_model_cached(model_path)
        st.session_state['model'] = model
        st.session_state['model_path'] = model_path
        st.sidebar.success(f"Model loaded and saved in session: {model_path}")
    except Exception as e:
        st.sidebar.error("Load failed: " + str(e))

# Show loaded model status
if 'model' in st.session_state:
    st.sidebar.info("Model in session: " + st.session_state.get('model_path', 'unknown'))
else:
    st.sidebar.warning("No model loaded. Click 'Load model' to load it into session.")

st.header("Predict (manual inputs)")
with st.expander("Show / Edit sample input form"):
    with st.form("input_form"):
        Age = st.number_input("Age", value=25)
        Daily_Screen_Time = st.number_input("Daily_Screen_Time(hrs)", value=2.5, format="%.2f")
        Sleep_Quality = st.number_input("Sleep_Quality(1-10)", value=7.0, format="%.2f")
        Stress_Level = st.number_input("Stress_Level(1-10)", value=4.0, format="%.2f")
        Days_Without_Social_Media = st.number_input("Days_Without_Social_Media", value=1)
        Exercise_Frequency = st.number_input("Exercise_Frequency(week)", value=2)
        Gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)
        Social_Media_Platform = st.text_input("Social_Media_Platform", value="Instagram")
        submitted = st.form_submit_button("Predict")

    if submitted:
        if 'model' not in st.session_state:
            st.warning("Please click 'Load model' first.")
        else:
            model = st.session_state['model']
            input_df = pd.DataFrame([{
                "Age": Age,
                "Daily_Screen_Time(hrs)": Daily_Screen_Time,
                "Sleep_Quality(1-10)": Sleep_Quality,
                "Stress_Level(1-10)": Stress_Level,
                "Days_Without_Social_Media": Days_Without_Social_Media,
                "Exercise_Frequency(week)": Exercise_Frequency,
                "Gender": Gender,
                "Social_Media_Platform": Social_Media_Platform
            }])
            try:
                pred = model.predict(input_df)[0]
                st.success(f"Predicted: {pred}")
                try:
                    proba = model.predict_proba(input_df)[0]
                    st.write("Prediction probabilities:", proba)
                except Exception:
                    # some models may not implement predict_proba
                    pass
            except Exception as e:
                st.error("Prediction failed: " + str(e))

st.markdown("---")
st.write("Selected model path:", model_path)
if 'model' in st.session_state:
    st.write("Model loaded from session:", st.session_state.get('model_path'))
