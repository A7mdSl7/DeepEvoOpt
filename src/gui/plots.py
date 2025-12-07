
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def plot_convergence(histories, optimizers):
    """
    Plots convergence of validation loss over iterations.
    histories: list of DataFrames
    optimizers: list of optimizer names corresponding to histories
    """
    if not histories:
        return None

    fig = go.Figure()

    for df, opt in zip(histories, optimizers):
        if df is None or df.empty:
            continue
            
        fig.add_trace(go.Scatter(
            x=df['iteration'],
            y=df['val_loss'],
            mode='lines+markers',
            name=opt,
            hovertemplate=f"<b>{opt}</b><br>Iter: %{{x}}<br>Msg: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(
        title="Convergence Plot (Validation Loss)",
        xaxis_title="Iteration",
        yaxis_title="Validation Loss",
        legend_title="Optimizer",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def plot_distribution(best_losses, optimizers):
    """
    Plots distribution of best validation losses (if multiple runs exist).
    Current implementation mocks multiple runs if only single run data is passed,
    or just shows a single point.
    """
    # Create a DataFrame for plotting
    data = []
    for loss, opt in zip(best_losses, optimizers):
        if loss is not None:
             data.append({'Optimizer': opt, 'Best Loss': loss})
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return None
        
    fig = px.box(df, x='Optimizer', y='Best Loss', points="all", title="Best Loss Distribution")
    fig.update_layout(template="plotly_white")
    return fig

def plot_confusion_matrix(cm, labels=None):
    """
    Plots confusion matrix heatmap.
    """
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
        
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    title="Confusion Matrix")
    fig.update_layout(template="plotly_white")
    return fig
