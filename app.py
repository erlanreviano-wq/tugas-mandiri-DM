import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")
if not MODEL_DIR.exists():
    st.error("Run main.py first to produce models in ./models")
    st.stop()

pipe_voting = joblib.load(MODEL_DIR / 'pipeline_voting.pkl')
pipe_rf = joblib.load(MODEL_DIR / 'pipeline_rf.pkl')
pipe_lr = joblib.load(MODEL_DIR / 'pipeline_lr.pkl')
feature_cols = pipe_voting.named_steps['preprocessor'].transformers_[0][2] + pipe_voting.named_steps['preprocessor'].transformers_[1][2]

st.title("Mental Health - Prediction (Head-to-Head)")
choice = st.sidebar.selectbox("Pilih model", ["Voting (RF+LR)", "RandomForest", "LogisticRegression"])

st.write("Masukkan fitur:")
with st.form('f'):
    inputs = {}
    for c in feature_cols:
        inputs[c] = st.text_input(c, "")
    submit = st.form_submit_button("Predict")

if submit:
    df_in = pd.DataFrame([inputs])
    for col in df_in.columns:
        try:
            df_in[col] = pd.to_numeric(df_in[col])
        except:
            pass
    if choice == "Voting (RF+LR)":
        pred = pipe_voting.predict(df_in)[0]
    elif choice == "RandomForest":
        pred = pipe_rf.predict(df_in)[0]
    else:
        pred = pipe_lr.predict(df_in)[0]
    st.write("Predicted:", int(pred) if isinstance(pred,(int,np.integer)) else str(pred))
