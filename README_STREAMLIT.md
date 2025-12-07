# DeepEvoOpt Streamlit Dashboard

This is an interactive dashboard for visualizing and controlling Deep Learning optimization experiments using Meta-Heuristic algorithms.

## Features
- **Convergence Plots**: Compare multiple optimizers (GA, PSO, GWO, etc.) on val_loss.
- **Summary KPIs**: Quick look at best performance, iteration counts, and stability.
- **Hyperparameter Inspection**: View and download the best hyperparameters found.
- **Run Experiments**: Light-weight UI to trigger new optimization runs.

## Running Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Dashboard**
   ```bash
   streamlit run app_streamlit.py
   ```

3. **Open Browser**
   The app will typically run at `http://localhost:8501`.

## Docker Deployment

1. **Build Image**
   ```bash
   docker build -t deepevoopt-dashboard .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 -v $(pwd)/results:/app/results deepevoopt-dashboard
   ```
   *Note: Mounting the `results` volume ensures logs persist outside the container.*

## Project Structure
- `app_streamlit.py`: Main entry point.
- `src/gui/`: Dashboard modules (data, plots, ui, run_control).
- `results/logs/`: Directory where CSV/JSON logs are stored.
