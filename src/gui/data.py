import os
import glob
import json
import pandas as pd
import streamlit as st

RESULTS_DIR = "results/logs"

def get_available_logs():
    """
    Scans the results directory and returns a dictionary of available experiments.
    Structure: {'cnn': ['ga', 'pso', ...], 'mlp': ['ga', ...]}
    """
    if not os.path.exists(RESULTS_DIR):
        return {'cnn': [], 'mlp': []}

    available = {'cnn': set(), 'mlp': set()}
    
    # Look for history files: {opt}_{model}_history.csv
    files = glob.glob(os.path.join(RESULTS_DIR, "*_*_history.csv"))
    
    for f in files:
        filename = os.path.basename(f)
        parts = filename.replace("_history.csv", "").split("_")
        if len(parts) >= 2:
            model = parts[-1] # Assumes model is the last part (e.g. ga_cnn)
            opt = "_".join(parts[:-1])
            
            if model in available:
                available[model].add(opt)
                
    return {k: list(v) for k, v in available.items()}

@st.cache_data
def load_history(optimizer, model):
    """
    Loads the history CSV for a specific optimizer and model.
    """
    file_path = os.path.join(RESULTS_DIR, f"{optimizer}_{model}_history.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading history for {optimizer} on {model}: {e}")
            return None
    return None

@st.cache_data
def load_best_params(optimizer, model):
    """
    Loads the best parameters JSON for a specific optimizer and model.
    """
    file_path = os.path.join(RESULTS_DIR, f"{optimizer}_{model}_best.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading best params for {optimizer} on {model}: {e}")
            return None
    return None

def load_user_file(uploaded_file):
    """
    Parses a user uploaded file (CSV or JSON).
    """
    if uploaded_file is None:
        return None
        
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return json.load(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}")
        return None
