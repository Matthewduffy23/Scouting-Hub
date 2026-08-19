"""Shared 'Download PNG' button for the matplotlib charts on the position pages.

Every chart on the position pages is a matplotlib figure rendered through st.pyplot,
so exporting is just savefig() into a buffer. Figures are built at dpi=100, so the
default dpi here (200) is the 2× export the visuals were designed to be shared at.

bbox_inches is deliberately left at None: several of these figures (the scatterplots
in particular) place labels by hand and "tight" re-crops the canvas, which moves the
labels off the points. Keep the saved canvas identical to the one on screen.
"""

from io import BytesIO

import streamlit as st

DEFAULT_DPI = 200
DEFAULT_LABEL = "⬇️ Download PNG"


def figure_png_bytes(fig, dpi: int = DEFAULT_DPI) -> bytes:
    """Render a matplotlib figure to PNG bytes at `dpi`, keeping its own colours."""
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_edgecolor(),
        bbox_inches=None,
    )
    buf.seek(0)
    return buf.getvalue()


def png_download_button(fig, file_name: str, label: str = DEFAULT_LABEL,
                        dpi: int = DEFAULT_DPI, key=None, help=None):
    """Draw a download button for `fig`. Never raises — a chart that can't be
    exported shows a caption instead of taking the whole page down."""
    if fig is None:
        return False
    try:
        data = figure_png_bytes(fig, dpi=dpi)
    except Exception as e:                      # noqa: BLE001 — export is best-effort
        st.caption(f"PNG export unavailable: {e}")
        return False
    return st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime="image/png",
        key=key,
        help=help,
    )
