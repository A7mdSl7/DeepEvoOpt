import streamlit as st
import time
import pandas as pd
from src.gui import ui, data, plots, run_control

# Page Config
st.set_page_config(
    page_title="DeepEvoOpt Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<h1 class='main-header'>🧬 DeepEvoOpt Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Compare meta-heuristic optimizers for Deep Learning models. (https://github.com/A7mdSl7/DeepEvoOpt)")
    st.markdown("---")

    # Sidebar
    controls = ui.render_sidebar()
    model_type = controls['model_type']
    selected_optimizers = controls['selected_optimizers']
    
    # --- Main Content ---
    
    # 1. Load Data
    histories = []
    valid_optimizers = []
    best_losses = [] # For distribution plot (mocked per run)
    
    for opt in selected_optimizers:
        df = data.load_history(opt, model_type)
        if df is not None:
            # Apply Smoothing
            if controls['smoothing'] > 1:
                df['val_loss'] = df['val_loss'].rolling(window=controls['smoothing'], min_periods=1).mean()
            
            histories.append(df)
            valid_optimizers.append(opt)
            best_losses.append(df['val_loss'].min())
            
    # Handle uploaded file
    if controls['user_data'] is not None:
        if isinstance(controls['user_data'], pd.DataFrame):
            histories.append(controls['user_data'])
            valid_optimizers.append(f"Upload: {controls['uploaded_filename']}")
            best_losses.append(controls['user_data']['val_loss'].min())

    # 2. Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary & Convergence", 
        "📈 Distribution", 
        "⚙️ Best Parameters", 
        "🧪 Sample Evaluation",
        "🚀 Run Experiments"
    ])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            ui.render_summary_table(histories, valid_optimizers)
            
        with col2:
            fig = plots.plot_convergence(histories, valid_optimizers)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select an optimizer with available data to see the convergence plot.")
                
    with tab2:
        if valid_optimizers:
            fig_dist = plots.plot_distribution(best_losses, valid_optimizers)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No data available for distribution plot.")

    with tab3:
        if valid_optimizers:
            selected_opt_params = st.selectbox("Select Optimizer to View Params", valid_optimizers)
            # Handle uploaded file vs standard
            if "Upload:" not in selected_opt_params:
                params = data.load_best_params(selected_opt_params, model_type)
                ui.render_best_params(params, selected_opt_params)
            else:
                st.info("Hyperparameters not available for uploaded history files yet.")
        else:
            st.info("Select optimizers to view their hyperparameters.")

    with tab4:
        st.markdown("### Sample Evaluation Results")
        st.write("Feature coming soon: Load saved checkpoint and run inference on test set.")
        # Placeholder for confusion matrix
        # cm = [[50, 2, 1], [3, 45, 5], [1, 4, 48]]
        # st.plotly_chart(plots.plot_confusion_matrix(cm), use_container_width=True)

    with tab5:
        st.markdown("### 🚀 Run New Experiment")
        st.warning("⚠️ Warning: Running experiments here is limited to short runs for demonstration.")
        
        c1, c2 = st.columns(2)
        with c1:
            run_opt = st.selectbox("Optimizer", ["ga", "pso", "gwo", "aco"], key="run_opt")
            run_pop_size = st.number_input("Population Size", min_value=2, max_value=20, value=5)
        with c2:
            run_model = st.selectbox("Model", ["cnn", "mlp"], key="run_model")
            run_max_iter = st.number_input("Max Iterations", min_value=1, max_value=10, value=3)
        
        if st.button("Start Experiment"):
            started = run_control.trigger_experiment(run_opt, run_model, run_pop_size, run_max_iter)
            if started:
                st.toast(f"Started {run_opt} on {run_model}!", icon="🚀")
            else:
                st.error("Experiment already running. Please wait.")
        
        # Live Logs
        st.subheader("Live Logs")
        log_container = st.empty()
        
        if run_control.is_running():
            with st.spinner("Experiment running..."):
                while run_control.is_running():
                    logs = run_control.get_latest_logs()
                    for log in logs:
                        st.text(log)
                    time.sleep(1)
                st.success("Experiment Completed!")

if __name__ == "__main__":
    main()
