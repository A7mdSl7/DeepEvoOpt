
import streamlit as st
import pandas as pd
from src.gui import data

def render_sidebar():
    """
    Renders the sidebar controls and returns selected options.
    """
    st.sidebar.title("DeepEvoOpt Dashboard")
    
    # Model Selection
    model_type = st.sidebar.selectbox("Select Model", ["cnn", "mlp"])
    
    # Optimizer Selection
    try:
        available = data.get_available_logs()
        available_opts = available.get(model_type, [])
    except Exception as e:
        st.sidebar.error(f"Error scanning logs: {e}")
        available_opts = []
    
    # Default to all available, or specific hardcoded list if empty
    default_opts = ['ga', 'pso', 'gwo', 'aco', 'firefly', 'abc', 'obc_woa', 'fcr', 'fcgwo']
    options = list(set(available_opts) | set(default_opts))
    options.sort()
    
    # Set default selection
    default_selection = [o for o in options if o in available_opts] if available_opts else []
    if not default_selection and options:
        default_selection = [options[0]]  # Select first one if no data available
    
    selected_optimizers = st.sidebar.multiselect(
        "Select Optimizers",
        options,
        default=default_selection,
        help="Select one or more optimizers to compare. Only optimizers with available data will show results."
    )
    
    # Show info about available data
    if available_opts:
        st.sidebar.success(f"✅ {len(available_opts)} optimizer(s) with data available")
        st.sidebar.caption(f"Available: {', '.join([o.upper() for o in sorted(available_opts)])}")
    else:
        st.sidebar.info("ℹ️ No data files found. Run experiments first or upload data.")
    
    # Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")
    smoothing = st.sidebar.slider("Smoothing (Rolling Mean)", 1, 10, 1)
    
    # Data Upload - Support multiple files
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload Data Files")
    
    with st.sidebar.expander("ℹ️ How to upload files", expanded=False):
        st.write("""
        **Supported formats:** CSV or JSON
        
        **Required columns:**
        - `iteration` (optional, will use index if missing)
        - `val_loss` (required)
        
        **You can upload multiple files** to compare different experiments.
        Each file will appear as a separate optimizer in the comparison.
        """)
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload History Files (CSV/JSON)", 
        type=['csv', 'json'],
        accept_multiple_files=True,
        help="You can upload multiple CSV/JSON files to compare them. Each file should contain 'iteration' and 'val_loss' columns."
    )
    
    user_data_list = []
    uploaded_filenames = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            data_result = data.load_user_file(uploaded_file)
            if data_result is not None:
                user_data_list.append(data_result)
                uploaded_filenames.append(uploaded_file.name)
            else:
                st.sidebar.warning(f"⚠️ Could not load: {uploaded_file.name}")
        
    return {
        "model_type": model_type,
        "selected_optimizers": selected_optimizers,
        "smoothing": smoothing,
        "user_data_list": user_data_list,
        "uploaded_filenames": uploaded_filenames
    }

def render_summary_table(histories, optimizers):
    """
    Renders a summary DataFrame comparing optimizers.
    """
    if not histories or not optimizers:
        st.info("No data available for summary. Please select optimizers with available data.")
        return

    summary = []
    best_overall_loss = float('inf')
    best_opt = None
    
    for df, opt in zip(histories, optimizers):
        if df is None or df.empty:
            continue
        if 'val_loss' not in df.columns:
            continue
            
        try:
            best_loss = float(df['val_loss'].min())
            best_idx = df['val_loss'].idxmin()
            
            if 'iteration' in df.columns:
                best_iter = int(df.loc[best_idx, 'iteration'])
            else:
                best_iter = int(best_idx)
            
            runtime = "N/A" # Placeholder if not in logs
            
            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_opt = opt
                
            summary.append({
                "Optimizer": opt,
                "Best Val Loss": f"{best_loss:.4f}",
                "Best Iteration": best_iter,
                "Runtime (s)": runtime
            })
        except Exception as e:
            st.warning(f"Error processing data for {opt}: {e}")
            continue
        
    if not summary:
        st.info("No valid data found in selected optimizers.")
        return

    st.subheader("📊 Summary KPIs")
    
    # Highlight best optimizer
    if best_opt:
        st.success(f"🏆 Best Performer: **{best_opt.upper()}** (Loss: {best_overall_loss:.4f})")
        
    df_summary = pd.DataFrame(summary)
    # Format the dataframe for better display
    st.dataframe(df_summary, use_container_width=True)

def render_best_params(params, optimizer):
    """
    Renders the best hyperparameters in a nice table.
    """
    if not params:
        st.warning(f"No best parameters found for {optimizer}.")
        return

    st.subheader(f"Best Hyperparameters: {optimizer.upper()}")
    
    # Create a cleaner display dictionary
    display_params = {k: v for k, v in params.items() if k not in ['val_loss', 'val_acc']}
    
    # Convert to DataFrame for display
    # Format values appropriately: keep numbers as numbers, convert others to string
    formatted_params = []
    for k, v in display_params.items():
        # Format numeric values nicely
        if isinstance(v, (int, float)):
            if isinstance(v, float):
                formatted_value = f"{v:.6f}" if abs(v) < 1 else f"{v:.4f}"
            else:
                formatted_value = str(v)
        else:
            formatted_value = str(v)
        formatted_params.append({'Hyperparameter': k, 'Value': formatted_value})
    
    df = pd.DataFrame(formatted_params)
    # All values are now strings, so Arrow serialization will work
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Download button
    import json as json_lib
    json_str = json_lib.dumps(params, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"{optimizer}_best_params.json",
        mime="application/json"
    )
