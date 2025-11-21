import streamlit as st
import sys, os
import joblib
import pandas as pd
import sklearn

st.title("DEBUG: Streamlit startup diagnostics")

st.subheader("Environment")
st.write("Python:", sys.version.splitlines()[0])
st.write("scikit-learn:", sklearn.__version__)
st.write("CWD:", os.getcwd())

st.subheader("Files in repo root")
root_files = os.listdir(".")
st.write(root_files)

# check expected model paths
expected_models = [
    "models/pipeline_voting.pkl",
    "models/pipeline_rf.pkl",
    "models/pipeline_lr.pkl"
]
st.subheader("Model file checks")
for p in expected_models:
    exists = os.path.exists(p)
    st.write(p, "exists:", exists)
    if exists:
        st.write("size (bytes):", os.path.getsize(p))

st.subheader("Try loading one model (safe)")
if st.button("Try load pipeline_voting"):
    p = "models/pipeline_voting.pkl"
    if not os.path.exists(p):
        st.error("File not found: " + p)
    else:
        try:
            m = joblib.load(p)
            st.success("Model loaded OK. Type: " + str(type(m)))
            # Print preprocessor info if available
            try:
                st.write("Pipeline steps:", m.named_steps.keys())
            except Exception:
                pass
        except Exception as e:
            st.exception(e)

st.write("If this shows an exception or missing files, fix the errors shown above.")
