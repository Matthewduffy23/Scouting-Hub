# app.py — Scouting Hub (Head Office)
# -----------------------------------
# Professional front page that routes to role apps in /pages.
# • Big navigation cards for Strikers / Attackers
# • Clean “FM-style” layout with subtle gradient, shadowed cards
# • Room for quick KPIs, recent activity, and notes
# • Works with st.switch_page() (Streamlit ≥ 1.27); falls back to page_link

import streamlit as st
from datetime import datetime

# ---------- Page config ----------
st.set_page_config(
    page_title="Scouting Hub — Head Office",
    page_icon="🏟️",
    layout="wide",
    menu_items={
        "About": "Internal Scouting Platform — Head Office",
    },
)

# ---------- Minimal brand CSS ----------
st.markdown(
    """
    <style>
      /* overall */
      .main {
        background: radial-gradient(1200px 600px at 20% 0%, #0f172a 0%, #0b1220 40%, #0b1020 100%);
        color: #e5e7eb;
      }
      section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(255,255,255,0.06);
      }
      /* headings */
      h1, h2, h3 { letter-spacing: 0.3px; }
      /* cards */
      .card {
        border-radius: 16px;
        padding: 20px 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 24px rgba(0,0,0,0.25);
      }
      .card:hover { transform: translateY(-1px); transition: transform .12s ease-out; }
      .navcard {
        border-radius: 16px;
        padding: 24px;
        background: linear-gradient(180deg, rgba(34,197,94,0.08), rgba(34,197,94,0.04));
        border: 1px solid rgba(34,197,94,0.25);
        box-shadow: 0 12px 24px rgba(0,0,0,0.25);
      }
      .muted { color: #9ca3af; }
      .pill {
        display:inline-block; padding:4px 10px; border-radius:999px;
        background: rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12);
        font-size: 12px; margin-right:8px; color:#e5e7eb;
      }
      .soft { color:#cbd5e1; }
      .hr { height:1px; background:rgba(255,255,255,0.08); margin:10px 0 14px 0; }
      .kpi {
        display:flex; flex-direction:column; gap:4px;
      }
      .kpi .label { font-size:12px; color:#9ca3af; }
      .kpi .value { font-size:24px; font-weight:700; color:#e5e7eb; }
      /* reduce default block padding a touch */
      div.block-container { padding-top: 36px; padding-bottom: 40px; }
      /* links color */
      a, a:visited { color:#86efac; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
col_logo, col_title = st.columns([1,6], vertical_alignment="center")
with col_logo:
    st.markdown("### 🏟️")
with col_title:
    st.markdown("## **Scouting Hub — Head Office**")
    st.caption("Internal scouting platform • Department access & quick status")

st.markdown("")

# ---------- Top KPIs (optional placeholders) ----------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="card kpi"><span class="label">Active Leagues</span><span class="value">32</span></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="card kpi"><span class="label">Players in Pool</span><span class="value">7,842</span></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="card kpi"><span class="label">Shortlist</span><span class="value">18</span></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="card kpi"><span class="label">Last Sync</span><span class="value">{datetime.utcnow().strftime("%d %b %Y")}</span></div>', unsafe_allow_html=True)

st.markdown("")

# ---------- Navigation row (FM-style "Departments") ----------
st.markdown("#### Departments")

left, right = st.columns(2, gap="large")

def goto(page_path: str):
    """Try st.switch_page if available, otherwise show a link."""
    try:
        st.switch_page(page_path)
    except Exception:
        st.page_link(page_path, label="Open page", icon="↗️")

with left:
    st.markdown(
        """
        <div class="navcard">
          <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:28px;">⚽</div>
            <div>
              <div style="font-weight:700; font-size:20px;">Strikers</div>
              <div class="muted">Finishing, xG/xGOT, box touches, aerial threat, movement profiles.</div>
            </div>
          </div>
          <div class="hr"></div>
          <div>
            <span class="pill">CF</span><span class="pill">Poacher</span><span class="pill">Target</span><span class="pill">Pressing 9</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Open Strikers Department", type="primary", use_container_width=True):
        goto("pages/01_Striker.py")

with right:
    st.markdown(
        """
        <div class="navcard">
          <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:28px;">🎯</div>
            <div>
              <div style="font-weight:700; font-size:20px;">Attackers</div>
              <div class="muted">Chance creation, carries, 1v1s, crossing, xA, progressive actions.</div>
            </div>
          </div>
          <div class="hr"></div>
          <div>
            <span class="pill">RW</span><span class="pill">LW</span><span class="pill">Inside Fwd</span><span class="pill">10 / SS</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Open Attackers Department", type="primary", use_container_width=True):
        goto("pages/02_Attacker.py")

st.markdown("")

# ---------- Tabs: quick notes / activity / documentation ----------
tabs = st.tabs(["📋 Desk Notes", "🕑 Recent Activity", "📚 Handbook"])

with tabs[0]:
    st.markdown(
        """
        <div class="card">
          <strong>Director’s notes</strong>
          <ul>
            <li>Summer shortlist to narrow from 18 → 10 by end of month.</li>
            <li>Weight league strength in role tables for EFL targets.</li>
            <li>Sync medical & character flags before final board deck.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
              <strong>Pipeline</strong>
              <ul class="soft">
                <li>🇵🇹 RW (U23) — video cut-ups requested</li>
                <li>🇩🇪 CF (Bosman) — agent call Friday</li>
                <li>🇧🇷 LW — visa feasibility check</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <strong>Recently viewed</strong>
              <ul class="soft">
                <li>01_Striker → Radar comparison</li>
                <li>02_Attacker → Similar players pool</li>
                <li>02_Attacker → Scatter: NPG vs xG</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[2]:
    st.markdown(
        """
        <div class="card">
          <strong>Usage</strong>
          <ol class="soft">
            <li>Open a department (Strikers / Attackers).</li>
            <li>Filter leagues, minutes, age and value bands in the sidebar.</li>
            <li>Use tables for Overalls, U-23, Expiring, Value band.</li>
            <li>Open a player to view profile radar, strengths/weaknesses, and role fit.</li>
          </ol>
          <div class="hr"></div>
          <strong>Data & Requirements</strong>
          <div class="soft">Shared files: <code>WORLDJUNE25.csv</code>, <code>requirements.txt</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Footer ----------
st.markdown("---")
st.caption("Scouting Department — Internal • © 2025")



