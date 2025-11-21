import streamlit as st
import joblib
import os
import pandas as pd

st.title("Mental Health - Prediction (Flexible loader)")

# possible model paths (first models/ then root)
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
    st.error("Model files not found. Expected one of:\n" + "\n".join(POSSIBLE))
    st.write("Files in repo root:", os.listdir("."))
    st.stop()

st.sidebar.write("Detected model files:")
for k,v in found.items():
    st.sidebar.write(f"- {k}  -> {v}")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

choice = st.sidebar.selectbox("Model to use", list(found.keys()))
model_path = found[choice]

if st.button("Load model"):
    try:
        model = load_model(model_path)
        st.success(f"Model loaded from {model_path}")
        st.write("Pipeline steps:", getattr(model, "named_steps", {}).keys())
    except Exception as e:
        st.exception(e)

st.header("Predict (manual inputs)")
if st.checkbox("Show sample input form"):
    with st.form("f"):
        age = st.number_input("Age", value=25)
        screen = st.number_input("Daily_Screen_Time(hrs)", value=2.5)
        sleep = st.number_input("Sleep_Quality(1-10)", value=7.0)
        stress = st.number_input("Stress_Level(1-10)", value=4.0)
        days_without = st.number_input("Days_Without_Social_Media", value=1)
        exercise = st.number_input("Exercise_Frequency(week)", value=2)
        gender = st.text_input("Gender", value="Female")
        platform = st.text_input("Social_Media_Platform", value="Instagram")
        submit = st.form_submit_button("Predict")

    if submit:
        if 'model' not in locals():
            st.warning("Please click 'Load model' first.")
        else:
            df = pd.DataFrame([{
                "Age": age,
                "Daily_Screen_Time(hrs)": screen,
                "Sleep_Quality(1-10)": sleep,
                "Stress_Level(1-10)": stress,
                "Days_Without_Social_Media": days_without,
                "Exercise_Frequency(week)": exercise,
                "Gender": gender,
                "Social_Media_Platform": platform
            }])
            try:
                pred = model.predict(df)[0]
                st.success(f"Predicted: {pred}")
            except Exception as e:
                st.exception(e)
