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
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return {'cnn': [], 'mlp': []}

    available = {'cnn': set(), 'mlp': set()}
    
    # Look for history files: {opt}_{model}_history.csv
    # Pattern examples: ga_cnn_history.csv, obc_woa_cnn_history.csv
    files = glob.glob(os.path.join(RESULTS_DIR, "*_*_history.csv"))
    
    for f in files:
        try:
            filename = os.path.basename(f)
            # Remove _history.csv suffix
            base_name = filename.replace("_history.csv", "")
            
            # Try to find model type (cnn or mlp) at the end
            if base_name.endswith("_cnn"):
                model = "cnn"
                opt = base_name[:-4]  # Remove "_cnn"
            elif base_name.endswith("_mlp"):
                model = "mlp"
                opt = base_name[:-4]  # Remove "_mlp"
            else:
                # Fallback: assume last part is model
                parts = base_name.split("_")
                if len(parts) >= 2:
                    model = parts[-1]
                    opt = "_".join(parts[:-1])
                else:
                    continue
            
            if model in available:
                available[model].add(opt)
        except Exception:
            # Skip files that don't match expected pattern
            continue
                
    return {k: sorted(list(v)) for k, v in available.items()}

@st.cache_data
def load_history(optimizer, model):
    """
    Loads the history CSV for a specific optimizer and model.
    """
    file_path = os.path.join(RESULTS_DIR, f"{optimizer}_{model}_history.csv")
    if not os.path.exists(file_path):
        # Return None silently - we'll handle the message in the main app
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Validate required columns
        if df.empty:
            return None
        if 'iteration' not in df.columns or 'val_loss' not in df.columns:
            # st.warning(f"History file for {optimizer} on {model} is missing required columns (iteration, val_loss).")
            return None
        return df
    except Exception as e:
        # Return None and let the main app handle the error message
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
    Returns DataFrame if successful, None otherwise.
    """
    if uploaded_file is None:
        return None
        
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Validate required columns
            if 'val_loss' not in df.columns:
                return None
            # If iteration column is missing, create it from index
            if 'iteration' not in df.columns:
                df['iteration'] = df.index
            return df
            
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # If it's a dict with history data, convert to DataFrame
                if 'iteration' in data and 'val_loss' in data:
                    # Both are lists/arrays
                    if isinstance(data['iteration'], list) and isinstance(data['val_loss'], list):
                        return pd.DataFrame(data)
                    else:
                        # Single values - create DataFrame with one row
                        return pd.DataFrame([data])
                elif 'history' in data:
                    # Nested history
                    return pd.DataFrame(data['history'])
                else:
                    # Try to convert dict to DataFrame
                    # Check if values are lists (like {'iteration': [1,2,3], 'val_loss': [0.5, 0.4, 0.3]})
                    if all(isinstance(v, list) for v in data.values() if v is not None):
                        return pd.DataFrame(data)
                    else:
                        # Best params format - not suitable for history
                        return None
            elif isinstance(data, list):
                # List of records
                df = pd.DataFrame(data)
                # Validate
                if 'val_loss' not in df.columns:
                    return None
                if 'iteration' not in df.columns:
                    df['iteration'] = df.index
                return df
            else:
                return None
                
    except Exception as e:
        # Don't show error here - let the main app handle it
        return None
