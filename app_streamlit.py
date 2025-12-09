import streamlit as st
import time
import pandas as pd
from src.gui import ui, data, plots, run_control

# Page Config
st.set_page_config(
    page_title="DeepEvoOpt Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styles
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def main():
    # Initialize session state for run_control
    run_control.init_session_state()

    # Header
    st.markdown(
        "<h1 class='main-header'>🧬 DeepEvoOpt Dashboard</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "Compare meta-heuristic optimizers for Deep Learning models. (https://github.com/A7mdSl7/DeepEvoOpt)"
    )
    st.markdown("---")

    # Sidebar
    try:
        controls = ui.render_sidebar()
        model_type = controls["model_type"]
        selected_optimizers = controls["selected_optimizers"]
    except Exception as e:
        st.error(f"Error loading sidebar: {e}")
        st.stop()

    # --- Main Content ---

    # 1. Load Data
    histories = []
    valid_optimizers = []
    best_losses = []  # For distribution plot (mocked per run)
    missing_data_optimizers = []  # Track optimizers without data

    if selected_optimizers:
        st.info(
            f"📋 Selected optimizers: {', '.join([opt.upper() for opt in selected_optimizers])}"
        )

        for opt in selected_optimizers:
            try:
                df = data.load_history(opt, model_type)
                if df is not None and not df.empty:
                    # Apply Smoothing
                    if controls["smoothing"] > 1:
                        df = df.copy()  # Avoid SettingWithCopyWarning
                        df["val_loss"] = (
                            df["val_loss"]
                            .rolling(window=controls["smoothing"], min_periods=1)
                            .mean()
                        )

                    histories.append(df)
                    valid_optimizers.append(opt)
                    if "val_loss" in df.columns:
                        best_losses.append(float(df["val_loss"].min()))
                else:
                    missing_data_optimizers.append(opt)
            except Exception as e:
                st.warning(f"⚠️ Could not load data for {opt}: {e}")
                missing_data_optimizers.append(opt)
                continue

        # Show message about missing data
        if missing_data_optimizers:
            with st.expander(
                f"⚠️ No data found for {len(missing_data_optimizers)} optimizer(s)",
                expanded=True,
            ):
                st.write(
                    f"**Missing data for:** {', '.join([opt.upper() for opt in missing_data_optimizers])}"
                )
                st.write("**To generate data:**")
                st.write("1. Go to the '🚀 Run Experiments' tab")
                st.write("2. Or run from command line:")
                for opt in missing_data_optimizers[:3]:  # Show first 3 examples
                    st.code(
                        f"python src/run_experiments.py --optimizer {opt} --model_type {model_type} --pop_size 10 --max_iter 5",
                        language="bash",
                    )

        if valid_optimizers:
            st.success(
                f"✅ Loaded data for: {', '.join([opt.upper() for opt in valid_optimizers])}"
            )
    else:
        st.info(
            "👈 Please select at least one optimizer from the sidebar to view results."
        )

    # Handle uploaded files (multiple files support)
    if controls["user_data_list"]:
        uploaded_count = len(controls["user_data_list"])
        st.success(f"📤 Successfully loaded {uploaded_count} uploaded file(s)")

        for idx, user_data in enumerate(controls["user_data_list"]):
            try:
                if isinstance(user_data, pd.DataFrame):
                    # Check if DataFrame has required columns
                    if "val_loss" in user_data.columns:
                        # Apply smoothing if needed
                        if controls["smoothing"] > 1:
                            user_data = user_data.copy()
                            user_data["val_loss"] = (
                                user_data["val_loss"]
                                .rolling(window=controls["smoothing"], min_periods=1)
                                .mean()
                            )

                        histories.append(user_data)
                        filename = (
                            controls["uploaded_filenames"][idx]
                            if idx < len(controls["uploaded_filenames"])
                            else f"uploaded_file_{idx+1}"
                        )
                        # Remove extension for cleaner display
                        display_name = (
                            filename.replace(".csv", "")
                            .replace(".json", "")
                            .replace("_history", "")
                        )
                        valid_optimizers.append(f"📤 {display_name}")
                        best_losses.append(float(user_data["val_loss"].min()))
                    else:
                        filename = (
                            controls["uploaded_filenames"][idx]
                            if idx < len(controls["uploaded_filenames"])
                            else f"file_{idx+1}"
                        )
                        st.warning(
                            f"⚠️ {filename} must contain 'val_loss' column. Skipping this file."
                        )
                else:
                    filename = (
                        controls["uploaded_filenames"][idx]
                        if idx < len(controls["uploaded_filenames"])
                        else f"file_{idx+1}"
                    )
                    st.warning(f"⚠️ {filename} is not a valid DataFrame. Skipping.")
            except Exception as e:
                filename = (
                    controls["uploaded_filenames"][idx]
                    if idx < len(controls["uploaded_filenames"])
                    else f"file_{idx+1}"
                )
                st.error(f"❌ Error processing {filename}: {e}")

    # 2. Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Summary & Convergence",
            "📈 Distribution",
            "⚙️ Best Parameters",
            "🧪 Sample Evaluation",
            "🚀 Run Experiments",
        ]
    )

    with tab1:
        if not valid_optimizers:
            if selected_optimizers:
                st.warning(
                    f"⚠️ No data available for selected optimizers: {', '.join([opt.upper() for opt in selected_optimizers])}. "
                    f"Please run experiments first or check the '🚀 Run Experiments' tab."
                )
            else:
                st.info(
                    "👈 Please select optimizers from the sidebar to view convergence plots and summary."
                )
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                ui.render_summary_table(histories, valid_optimizers)

            with col2:
                fig = plots.plot_convergence(histories, valid_optimizers)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No valid convergence data available. Make sure the data contains 'iteration' and 'val_loss' columns."
                    )

    with tab2:
        if valid_optimizers and best_losses:
            fig_dist = plots.plot_distribution(best_losses, valid_optimizers)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("Could not generate distribution plot. Check your data format.")
        else:
            st.info(
                "👈 Please select optimizers from the sidebar to view best loss comparison."
            )

    with tab3:
        if valid_optimizers:
            selected_opt_params = st.selectbox(
                "Select Optimizer to View Params",
                valid_optimizers,
                help="Select an optimizer to view its best hyperparameters",
            )
            # Handle uploaded file vs standard
            if "📤" not in selected_opt_params and "Upload:" not in selected_opt_params:
                # Standard optimizer - try to load best params
                params = data.load_best_params(selected_opt_params, model_type)
                if params:
                    ui.render_best_params(params, selected_opt_params)
                else:
                    st.warning(
                        f"No best parameters file found for {selected_opt_params}. Run the experiment first to generate parameters."
                    )
            else:
                # Uploaded file - hyperparameters not available
                st.info(
                    "ℹ️ Hyperparameters not available for uploaded history files. "
                    "Uploaded files only contain convergence history (iteration, val_loss). "
                    "To view hyperparameters, use standard optimizer results from experiments."
                )
        else:
            st.info(
                "👈 Please select optimizers from the sidebar to view their hyperparameters."
            )

    with tab4:
        st.markdown("### Sample Evaluation Results")
        st.write(
            "Feature coming soon: Load saved checkpoint and run inference on test set."
        )
        # Placeholder for confusion matrix
        # cm = [[50, 2, 1], [3, 45, 5], [1, 4, 48]]
        # st.plotly_chart(plots.plot_confusion_matrix(cm), use_container_width=True)

    with tab5:
        st.markdown("### 🚀 Run New Experiment")
        st.info(
            "💡 Run experiments directly from the dashboard. Results will be saved automatically and appear in other tabs."
        )

        c1, c2 = st.columns(2)
        with c1:
            run_opt = st.selectbox(
                "Optimizer",
                [
                    "ga",
                    "pso",
                    "gwo",
                    "aco",
                    "firefly",
                    "abc",
                    "obc_woa",
                    "fcr",
                    "fcgwo",
                ],
                key="run_opt",
            )
            run_pop_size = st.number_input(
                "Population Size",
                min_value=2,
                max_value=50,
                value=10,
                help="Number of solutions in each generation",
            )
        with c2:
            run_model = st.selectbox("Model", ["cnn", "mlp"], key="run_model")
            run_max_iter = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=50,
                value=5,
                help="Maximum number of optimization iterations",
            )

        # Check if experiment is running
        is_running = run_control.is_running()

        # Status indicator
        col_status1, col_status2 = st.columns([3, 1])
        with col_status1:
            if is_running:
                st.info("⏳ **Experiment is currently running. Please wait...**")
            else:
                st.success("✅ **Ready to run experiments**")

        with col_status2:
            if st.button(
                "🔄 Refresh Status", help="Refresh to check experiment status"
            ):
                st.rerun()

        # Start button
        if st.button(
            "🚀 Start Experiment",
            disabled=is_running,
            type="primary",
            use_container_width=True,
        ):
            started = run_control.trigger_experiment(
                run_opt, run_model, run_pop_size, run_max_iter
            )
            if started:
                st.success(f"✅ Started {run_opt.upper()} on {run_model.upper()}!")
                st.info(
                    "💡 The experiment is running in the background. Check the logs below and refresh the page to see updates."
                )
                st.rerun()
            else:
                st.error(
                    "❌ Experiment already running. Please wait for it to complete."
                )

        st.markdown("---")

        # Live Logs Section
        st.subheader("📋 Experiment Logs")

        # Clear logs button
        col_log1, col_log2 = st.columns([4, 1])
        with col_log2:
            if st.button("🗑️ Clear Logs"):
                run_control.clear_logs()
                st.rerun()

        # Auto-refresh option
        auto_refresh = st.checkbox(
            "🔄 Auto-refresh logs (every 2 seconds)", value=is_running
        )

        # Get and display logs
        logs = run_control.get_latest_logs()

        if logs:
            # Show last 100 lines
            log_text = "\n".join(logs[-100:])
            st.text_area(
                "Output",
                log_text,
                height=400,
                disabled=True,
                key="experiment_logs_display",
            )

            # Show log count
            st.caption(f"📊 Total log lines: {len(logs)} | Showing last 100 lines")
        else:
            if is_running:
                st.info(
                    "⏳ Waiting for experiment output... The experiment may take a few moments to start."
                )
            else:
                st.info(
                    "📝 No experiment logs yet. Start an experiment to see output here."
                )

        # Auto-refresh if enabled and running
        if auto_refresh and is_running:
            time.sleep(2)
            st.rerun()

        # Status message
        if not is_running and logs:
            st.success(
                "✅ Experiment completed! Check the logs above. Go to other tabs to view results."
            )

        # Instructions
        with st.expander("ℹ️ How to use", expanded=False):
            st.markdown(
                """
            **Steps:**
            1. Select optimizer, model type, population size, and max iterations
            2. Click "🚀 Start Experiment" button
            3. Watch the logs in real-time (enable auto-refresh for live updates)
            4. Once complete, go to other tabs to view results
            
            **Tips:**
            - Start with small values (pop_size=5, max_iter=3) for quick testing
            - Larger values take longer but give better results
            - Results are automatically saved to `results/logs/`
            - You can compare results in the "Summary & Convergence" tab
            """
            )


if __name__ == "__main__":
    main()
