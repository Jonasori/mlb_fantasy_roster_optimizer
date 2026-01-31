"""
Reusable UI components for the Streamlit dashboard.
"""

import io

import matplotlib.pyplot as plt
import streamlit as st


def display_figure(fig: plt.Figure, width: int = 400) -> None:
    """
    Display a matplotlib figure at a specific pixel width.

    This gives precise control over display size while maintaining
    high resolution from the underlying figure.

    Args:
        fig: matplotlib Figure object
        width: display width in pixels (default 400)
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    st.image(buf, width=width)
    plt.close(fig)
