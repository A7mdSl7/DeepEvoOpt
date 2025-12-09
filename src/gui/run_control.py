
import subprocess
import threading
import queue
import time
import sys
import os
import streamlit as st

def init_session_state():
    """
    Initialize session state variables if they don't exist.
    This should be called at the start of the app.
    """
    if 'experiment_logs' not in st.session_state:
        st.session_state.experiment_logs = []
    if 'experiment_process' not in st.session_state:
        st.session_state.experiment_process = None
    if 'experiment_running' not in st.session_state:
        st.session_state.experiment_running = False

# Initialize on import
init_session_state()

def run_experiment_thread(optimizer, model, pop_size, max_iter):
    """
    Worker function to run the experiment in a separate process.
    """
    # Ensure session state is initialized
    init_session_state()
    
    try:
        # Add initial messages
        st.session_state.experiment_logs.append(f"🚀 Starting {optimizer.upper()} on {model.upper()}...")
        st.session_state.experiment_logs.append(f"📊 Parameters: pop_size={pop_size}, max_iter={max_iter}")
        st.session_state.experiment_logs.append("─" * 50)
        
        # Construct command
        cmd = [
            sys.executable, "src/run_experiments.py",
            "--optimizer", optimizer,
            "--model_type", model,
            "--pop_size", str(pop_size),
            "--max_iter", str(max_iter)
        ]
        
        # Use Popen to capture stdout/stderr in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.getcwd()  # Use current working directory
        )
        
        st.session_state.experiment_process = process
        st.session_state.experiment_running = True
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if line:  # Only add non-empty lines
                    st.session_state.experiment_logs.append(line)
                    # Keep only last 1000 lines to avoid memory issues
                    if len(st.session_state.experiment_logs) > 1000:
                        st.session_state.experiment_logs = st.session_state.experiment_logs[-1000:]
        
        process.stdout.close()
        return_code = process.wait()
        
        st.session_state.experiment_logs.append("─" * 50)
        if return_code == 0:
            st.session_state.experiment_logs.append("✅ DONE: Experiment finished successfully!")
            st.session_state.experiment_logs.append(f"📁 Results saved to: results/logs/{optimizer}_{model}_*.csv/json")
        else:
            st.session_state.experiment_logs.append(f"❌ ERROR: Experiment failed with code {return_code}.")
            st.session_state.experiment_logs.append("💡 Check the error messages above for details.")
            
    except Exception as e:
        st.session_state.experiment_logs.append(f"❌ ERROR: Failed to launch experiment: {str(e)}")
        st.session_state.experiment_logs.append("💡 Make sure all dependencies are installed and the script path is correct.")
        import traceback
        st.session_state.experiment_logs.append(f"Traceback: {traceback.format_exc()}")
    finally:
        st.session_state.experiment_running = False
        st.session_state.experiment_process = None

def trigger_experiment(optimizer, model, pop_size, max_iter):
    """
    Launches a new experiment in a background thread.
    Returns True if started, False if busy.
    """
    # Ensure session state is initialized
    init_session_state()
    
    # Check if already running
    if st.session_state.experiment_running:
        if st.session_state.experiment_process is not None:
            try:
                poll_result = st.session_state.experiment_process.poll()
                if poll_result is None:  # Still running
                    return False
            except AttributeError:
                # Process is invalid, reset
                st.session_state.experiment_running = False
                st.session_state.experiment_process = None
    
    # Clear old logs
    st.session_state.experiment_logs = []
    
    # Start thread
    thread = threading.Thread(
        target=run_experiment_thread,
        args=(optimizer, model, pop_size, max_iter),
        daemon=True
    )
    thread.start()
    
    # Give thread a moment to start
    time.sleep(0.5)
    
    return True

def get_latest_logs():
    """
    Returns all logs from session state.
    """
    # Ensure session state is initialized
    init_session_state()
    return st.session_state.experiment_logs.copy()

def is_running():
    """
    Checks if an experiment is currently running.
    """
    # Ensure session state is initialized
    init_session_state()
    
    if not st.session_state.experiment_running:
        return False
    
    if st.session_state.experiment_process is None:
        st.session_state.experiment_running = False
        return False
    
    # Check if process is still alive
    try:
        poll_result = st.session_state.experiment_process.poll()
        if poll_result is not None:
            # Process has finished
            st.session_state.experiment_running = False
            st.session_state.experiment_process = None
            return False
    except AttributeError:
        # Process object is invalid
        st.session_state.experiment_running = False
        st.session_state.experiment_process = None
        return False
    
    return True

def clear_logs():
    """
    Clear experiment logs.
    """
    # Ensure session state is initialized
    init_session_state()
    st.session_state.experiment_logs = []
