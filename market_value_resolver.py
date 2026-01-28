import pandas as pd
import streamlit as st
from fotmob_value_utils import best_available_market_value_eur

@st.cache_data(show_spinner=False)
def apply_smart_market_value(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrites df['Market value'] with:
    FotMob value if found, else CSV value.
    """
    df = df_in.copy()
    if "Market value" not in df.columns:
        df["Market value"] = None

    df["Market value"] = df.apply(
        lambda r: best_available_market_value_eur(
            player=str(r.get("Player", "")),
            team=str(r.get("Team", "")),
            csv_market_value=r.get("Market value", None),
        ),
        axis=1
    )
    return df
