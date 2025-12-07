import sys
import os
import traceback

# Add root to path
sys.path.append(os.getcwd())

def test_imports():
    print("Testing imports...")
    try:
        import streamlit
        import plotly
        import pandas
        from src.gui import data, plots, ui, run_control
        print("Imports successful.")
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def test_data_loading():
    print("Testing data loading...")
    try:
        from src.gui import data
        # Mock results dir if needed or just check empty return
        available = data.get_available_logs()
        print(f"Available logs found: {available}")
        print("Data loading check passed.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_imports()
        test_data_loading()
        print("All checks passed!")
    except Exception as e:
        print(f"Unexpected error: {e}")
