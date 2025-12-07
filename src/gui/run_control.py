
import subprocess
import threading
import queue
import time
import sys
import streamlit as st

# Global queue for experiment output
output_queue = queue.Queue()
experiment_process = None

def run_experiment_thread(optimizer, model, pop_size, max_iter):
    """
    Worker function to run the experiment in a separate process.
    """
    global experiment_process
    
    # Construct command
    cmd = [
        sys.executable, "src/run_experiments.py",
        "--optimizer", optimizer,
        "--model_type", model,
        "--pop_size", str(pop_size),
        "--max_iter", str(max_iter)
    ]
    
    try:
        # Use Popen to capture stdout/stderr in real-time
        experiment_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=None # Uses current working directory
        )
        
        # Read output line by line
        for line in iter(experiment_process.stdout.readline, ''):
            if line:
                output_queue.put(line.strip())
        
        experiment_process.stdout.close()
        return_code = experiment_process.wait()
        
        if return_code == 0:
            output_queue.put("DONE: Experiment finished successfully.")
        else:
            output_queue.put(f"ERROR: Experiment failed with code {return_code}.")
            
    except Exception as e:
        output_queue.put(f"ERROR: Failed to launch experiment: {str(e)}")
    finally:
        experiment_process = None

def trigger_experiment(optimizer, model, pop_size, max_iter):
    """
    Launches a new experiment in a background thread.
    Returns True if started, False if busy.
    """
    global experiment_process
    
    if experiment_process is not None and experiment_process.poll() is None:
        return False # Busy
        
    thread = threading.Thread(
        target=run_experiment_thread,
        args=(optimizer, model, pop_size, max_iter),
        daemon=True
    )
    thread.start()
    return True

def get_latest_logs():
    """
    Returns all new logs from the queue.
    """
    logs = []
    while not output_queue.empty():
        try:
            logs.append(output_queue.get_nowait())
        except queue.Empty:
            break
    return logs

def is_running():
    global experiment_process
    return experiment_process is not None and experiment_process.poll() is None
