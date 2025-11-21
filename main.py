import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import re

# ---------- Config ----------
st.set_page_config(page_title="Mental Health - Prediction", layout="wide")
st.title("Mental Health - Prediction (Robust Loader)")

# Path to dataset uploaded earlier (used for reference only)
DATA_PATH = "/mnt/data/Mental_Health_and_Social_Media_Balance_Dataset.csv"

# Possible model file locations to look for (first models/, then repo root)
POSSIBLE_MODEL_PATHS = [
    "models/pipeline_voting.pkl",
    "models/pipeline_rf.pkl",
    "models/pipeline_lr.pkl",
    "pipeline_voting.pkl",
    "pipeline_rf.pkl",
    "pipeline_lr.pkl"
]

# ---------- Helpers ----------
def find_models():
    found = {}
    for p in POSSIBLE_MODEL_PATHS:
        if os.path.exists(p):
            found[os.path.basename(p)] = p
    return found

@st.cache_resource
def load_model_cached(path):
    """Load model from disk with caching to avoid repeated IO."""
    return joblib.load(path)

def parse_missing_columns_from_msg(msg: str):
    """
    Try to parse missing column names from exception message.
    Returns set of column names or empty set.
    Handles messages like:
      "columns are missing: {'User_ID'}"
    """
    # Try to find braces {...}
    m = re.search(r"columns are missing:\s*(\{.*\})", msg)
    if not m:
        # alternative phrasing
        m = re.search(r"Missing columns?[:\s]*([\[\{].*[\]\}])", msg, re.IGNORECASE)
    if not m:
        return set()
    s = m.group(1)
    # remove surrounding braces/brackets and split by commas safely
    s2 = s.strip().lstrip("{[").rstrip("]}").strip()
    if not s2:
        return set()
    # split items and clean quotes/spaces
    items = [it.strip().strip("\"'") for it in re.split(r",\s*", s2)]
    return set([it for it in items if it])

def fill_missing_columns(df: pd.DataFrame, missing: set):
    """
    Fill missing columns in dataframe with sensible defaults:
    - If column looks numeric (by name), fill 0
    - Else fill empty string
    """
    # heuristic numeric column names from this dataset
    numeric_like = {
        "Age",
        "Daily_Screen_Time(hrs)",
        "Sleep_Quality(1-10)",
        "Stress_Level(1-10)",
        "Days_Without_Social_Media",
        "Exercise_Frequency(week)"
    }
    for col in missing:
        if col in numeric_like:
            df[col] = 0
        else:
            # default to empty string for categorical / id-like
            df[col] = ""
    # ensure column order deterministic
    return df

# ---------- UI: Sidebar ----------
found_models = find_models()
if not found_models:
    st.sidebar.error("No model files found in repo. Expected one of:\n" + "\n".join(POSSIBLE_MODEL_PATHS))
    st.sidebar.write("Files in repo root:", os.listdir("."))
    st.stop()

st.sidebar.header("Detected model files")
for name, p in found_models.items():
    st.sidebar.write(f"- {name}  →  {p}")

# default choice prefers pipeline_voting if present
default_name = "pipeline_voting.pkl" if "pipeline_voting.pkl" in found_models else list(found_models.keys())[0]
model_choice = st.sidebar.selectbox("Model to use", list(found_models.keys()), index=list(found_models.keys()).index(default_name))
model_path = found_models[model_choice]

# Load model on demand and store to session_state
if st.sidebar.button("Load model"):
    try:
        model = load_model_cached(model_path)
        st.session_state['model'] = model
        st.session_state['model_path'] = model_path
        st.sidebar.success(f"Model loaded to session: {os.path.basename(model_path)}")
    except Exception as e:
        st.sidebar.error("Failed to load model: " + str(e))

# show model status
if 'model' in st.session_state:
    st.sidebar.info("Model in session: " + os.path.basename(st.session_state.get('model_path', 'unknown')))
else:
    st.sidebar.warning("No model loaded. Click 'Load model' to load it into session.")

st.sidebar.markdown("---")
st.sidebar.write("Dataset path (reference):")
st.sidebar.write(DATA_PATH if os.path.exists(DATA_PATH) else "Dataset not found in runtime (this is optional).")

# ---------- UI: Main form ----------
st.header("Predict (manual inputs)")
st.write("Fill the form below then click **Predict**. Make sure to click **Load model** first.")

with st.form("input_form"):
    # include User_ID because pipeline expects it
    User_ID = st.text_input("User_ID", value="")
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
        # Build dataframe using expected column names (as used during training)
        input_df = pd.DataFrame([{
            "User_ID": User_ID,
            "Age": Age,
            "Daily_Screen_Time(hrs)": Daily_Screen_Time,
            "Sleep_Quality(1-10)": Sleep_Quality,
            "Stress_Level(1-10)": Stress_Level,
            "Days_Without_Social_Media": Days_Without_Social_Media,
            "Exercise_Frequency(week)": Exercise_Frequency,
            "Gender": Gender,
            "Social_Media_Platform": Social_Media_Platform
        }])

        # Try predict; on missing-column error, auto-fill missing and retry once
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Predicted: {pred}")
            # show probabilities if available
            try:
                proba = model.predict_proba(input_df)[0]
                st.write("Prediction probabilities:", proba)
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            missing = parse_missing_columns_from_msg(msg)
            if missing:
                st.warning(f"Detected missing columns required by the model: {missing}. Filling with defaults and retrying...")
                input_df = fill_missing_columns(input_df, missing)
                try:
                    pred = model.predict(input_df)[0]
                    st.success(f"Predicted (after auto-fill): {pred}")
                    try:
                        proba = model.predict_proba(input_df)[0]
                        st.write("Prediction probabilities:", proba)
                    except Exception:
                        pass
                except Exception as e2:
                    st.error("Prediction retry failed: " + str(e2))
            else:
                # unknown error - display it
                st.error("Prediction failed: " + msg)

st.markdown("---")
st.write("Selected model path:", model_path)
if 'model' in st.session_state:
    st.write("Model loaded from session:", st.session_state.get('model_path'))
else:
    st.write("No model loaded in session.")
