import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

def generate_heatmap(scores: list, labels: list) -> go.Figure:
    """
    Generates a heatmap visualization of similarity scores.

    Args:
        scores (list): A list of similarity scores.
        labels (list): A list of labels for the x-axis (corresponding to sections).

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly heatmap figure object.
    """
    logger.info("Generating heatmap")
    fig = go.Figure(data=go.Heatmap(z=[scores], x=labels, y=["Similarity Score"],
                                   colorscale='Viridis'))
    fig.update_layout(title="Section Similarity Scores")
    return fig
