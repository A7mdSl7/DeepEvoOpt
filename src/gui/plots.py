
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
    if not histories or not optimizers:
        return None

    fig = go.Figure()
    has_data = False

    for df, opt in zip(histories, optimizers):
        if df is None or df.empty:
            continue
        if 'iteration' not in df.columns or 'val_loss' not in df.columns:
            continue
            
        has_data = True
        fig.add_trace(go.Scatter(
            x=df['iteration'],
            y=df['val_loss'],
            mode='lines+markers',
            name=opt,
            hovertemplate=f"<b>{opt}</b><br>Iter: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>"
        ))

    if not has_data:
        return None

    fig.update_layout(
        title="Convergence Plot (Validation Loss)",
        xaxis_title="Iteration",
        yaxis_title="Validation Loss",
        legend_title="Optimizer",
        hovermode="x unified",
        template="plotly_white",
        height=500
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
        if loss is not None and not pd.isna(loss):
             data.append({'Optimizer': opt, 'Best Loss': float(loss)})
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return None
    
    # Use bar chart instead of box plot for single values
    fig = px.bar(df, x='Optimizer', y='Best Loss', title="Best Loss Comparison", 
                 color='Best Loss', color_continuous_scale='RdYlGn_r')
    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Optimizer",
        yaxis_title="Best Validation Loss"
    )
    fig.update_traces(texttemplate='%{y:.4f}', textposition='outside')
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
