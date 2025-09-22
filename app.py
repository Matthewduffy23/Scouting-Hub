Put this in Scouting-Hub/app.py:

# app.py — Hub / Landing page
import streamlit as st

st.set_page_config(page_title="Scouting Hub", layout="wide")

st.title("🌐 Scouting Hub")
st.markdown("""
Welcome — this repo contains multiple scouting apps.

**How to use**
- Open the left sidebar and click **Pages** to choose:
  - `01_Striker`  — Striker-focused analyst
  - `02_Attacker`   — Attacker-focused analyst

The file `WORLDJUNE25.csv` and `requirements.txt` are shared by all pages.
""")

st.info("If you want this hub to redirect or show quick KPIs, edit this file.")


Save it as app.py (root of repo, not inside pages/).