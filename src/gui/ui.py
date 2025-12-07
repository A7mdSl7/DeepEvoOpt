
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
    available = data.get_available_logs()
    available_opts = available.get(model_type, [])
    
    # Default to all available, or specific hardcoded list if empty
    default_opts = ['ga', 'pso', 'gwo', 'aco', 'firefly', 'abc', 'obc_woa', 'fcr', 'fcgwo']
    options = list(set(available_opts) | set(default_opts))
    options.sort()
    
    selected_optimizers = st.sidebar.multiselect(
        "Select Optimizers",
        options,
        default=[o for o in options if o in available_opts]
    )
    
    # Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")
    smoothing = st.sidebar.slider("Smoothing (Rolling Mean)", 1, 10, 1)
    
    # Data Upload
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload History (CSV/JSON)", type=['csv', 'json'])
    
    user_data = None
    if uploaded_file:
        user_data = data.load_user_file(uploaded_file)
        
    return {
        "model_type": model_type,
        "selected_optimizers": selected_optimizers,
        "smoothing": smoothing,
        "user_data": user_data,
        "uploaded_filename": uploaded_file.name if uploaded_file else None
    }

def render_summary_table(histories, optimizers):
    """
    Renders a summary DataFrame comparing optimizers.
    """
    if not histories:
        st.info("No data available for summary.")
        return

    summary = []
    best_overall_loss = float('inf')
    best_opt = None
    
    for df, opt in zip(histories, optimizers):
        if df is None or df.empty:
            continue
            
        best_loss = df['val_loss'].min()
        best_iter = df.loc[df['val_loss'].idxmin(), 'iteration']
        runtime = "N/A" # Placeholder if not in logs
        
        if best_loss < best_overall_loss:
            best_overall_loss = best_loss
            best_opt = opt
            
        summary.append({
            "Optimizer": opt,
            "Best Val Loss": best_loss,
            "Best Iteration": int(best_iter),
            "Runtime (s)": runtime
        })
        
    if not summary:
        return

    st.subheader("Summary KPIs")
    
    # Highlight best optimizer
    if best_opt:
        st.success(f"🏆 Best Performer: **{best_opt.upper()}** (Loss: {best_overall_loss:.4f})")
        
    df_summary = pd.DataFrame(summary)
    st.dataframe(df_summary.style.highlight_min(subset=['Best Val Loss'], color='#d1e7dd'))

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
    df = pd.DataFrame(list(display_params.items()), columns=['Hyperparameter', 'Value'])
    st.table(df)
    
    # Download button
    json_str = str(params).replace("'", '"')
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"{optimizer}_best_params.json",
        mime="application/json"
    )
