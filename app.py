import streamlit as st
import requests
import time
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Breast Cancer AI — Random Forest",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #050d1a 0%, #0a1628 40%, #071020 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1e 0%, #050c18 100%) !important;
    border-right: 1px solid rgba(16, 185, 129, 0.15);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(59,130,246,0.08) 50%, rgba(139,92,246,0.06) 100%);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 24px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🧬';
    position: absolute;
    right: 48px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.12;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 14px;
    letter-spacing: 0.05em;
}

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(10,16,32,0.9));
    border: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    transition: border-color 0.3s, transform 0.2s;
}
.stat-card:hover {
    border-color: rgba(16, 185, 129, 0.4);
    transform: translateY(-2px);
}
.stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(90deg, #34d399, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-lbl { font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }

/* Section label */
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem; font-weight: 600;
    color: #34d399; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 20px 0 10px 0;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(16,185,129,0.3), transparent);
}

/* Input styling */
.stNumberInput input {
    background: rgba(8, 15, 30, 0.95) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
}
.stNumberInput input:focus {
    border-color: rgba(16, 185, 129, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(10, 18, 35, 0.8) !important;
    border: 1px solid rgba(16, 185, 129, 0.12) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; padding: 16px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    width: 100% !important; letter-spacing: 0.04em !important;
    box-shadow: 0 4px 24px rgba(5, 150, 105, 0.35) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(5, 150, 105, 0.55) !important;
}

/* Result cards */
.card-malignant {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(153,27,27,0.05));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 20px; padding: 32px; text-align: center;
    animation: glowRed 2.5s ease-in-out infinite;
}
.card-benign {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(4,120,87,0.05));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 20px; padding: 32px; text-align: center;
    animation: glowGreen 2.5s ease-in-out infinite;
}
@keyframes glowRed {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50% { box-shadow: 0 0 28px 6px rgba(239,68,68,0.12); }
}
@keyframes glowGreen {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
    50% { box-shadow: 0 0 28px 6px rgba(16,185,129,0.12); }
}
.card-icon { font-size: 3.2rem; margin-bottom: 8px; }
.card-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
.card-sub { color: #94a3b8; font-size: 0.9rem; }
.card-risk { font-size: 0.85rem; font-weight: 600; margin-top: 10px; padding: 5px 16px; border-radius: 20px; display: inline-block; }

/* Mini prob cards */
.prob-card {
    background: rgba(8,15,30,0.9);
    border-radius: 12px; padding: 16px; text-align: center;
    border: 1px solid rgba(16,185,129,0.1);
}
.prob-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; }
.prob-lbl { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 3px; }

/* Sidebar info */
.info-pill {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 10px; padding: 14px 16px;
    font-size: 0.83rem; color: #64748b; margin-bottom: 12px;
    line-height: 1.6;
}

/* Tree visualization */
.tree-stat {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid rgba(16,185,129,0.08);
    font-size: 0.83rem;
}
.tree-stat-label { color: #64748b; }
.tree-stat-val { color: #34d399; font-weight: 600; font-family: 'Space Grotesk', sans-serif; }

hr { border-color: rgba(16,185,129,0.08) !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

FEATURES = {
    "🔵 Mean Values": [
        ("mean_radius","Mean Radius",6.9,28.1,14.1),
        ("mean_texture","Mean Texture",9.7,39.3,19.3),
        ("mean_perimeter","Mean Perimeter",43.8,188.5,92.0),
        ("mean_area","Mean Area",143.5,2501.0,654.9),
        ("mean_smoothness","Mean Smoothness",0.05,0.16,0.096),
        ("mean_compactness","Mean Compactness",0.02,0.35,0.104),
        ("mean_concavity","Mean Concavity",0.0,0.43,0.089),
        ("mean_concave_points","Mean Concave Points",0.0,0.20,0.049),
        ("mean_symmetry","Mean Symmetry",0.10,0.30,0.181),
        ("mean_fractal_dimension","Mean Fractal Dim.",0.05,0.10,0.063),
    ],
    "🟡 Error Values": [
        ("radius_error","Radius Error",0.11,2.87,0.41),
        ("texture_error","Texture Error",0.36,4.88,1.22),
        ("perimeter_error","Perimeter Error",0.76,21.98,2.87),
        ("area_error","Area Error",6.8,542.2,40.3),
        ("smoothness_error","Smoothness Error",0.001,0.031,0.007),
        ("compactness_error","Compactness Error",0.002,0.135,0.025),
        ("concavity_error","Concavity Error",0.0,0.396,0.032),
        ("concave_points_error","Concave Pts Error",0.0,0.053,0.012),
        ("symmetry_error","Symmetry Error",0.007,0.079,0.021),
        ("fractal_dimension_error","Fractal Dim. Error",0.0008,0.030,0.004),
    ],
    "🔴 Worst Values": [
        ("worst_radius","Worst Radius",7.9,36.0,16.3),
        ("worst_texture","Worst Texture",12.0,49.5,25.7),
        ("worst_perimeter","Worst Perimeter",50.4,251.2,107.3),
        ("worst_area","Worst Area",185.2,4254.0,880.6),
        ("worst_smoothness","Worst Smoothness",0.07,0.22,0.132),
        ("worst_compactness","Worst Compactness",0.027,1.058,0.254),
        ("worst_concavity","Worst Concavity",0.0,1.252,0.272),
        ("worst_concave_points","Worst Concave Pts",0.0,0.291,0.115),
        ("worst_symmetry","Worst Symmetry",0.15,0.66,0.290),
        ("worst_fractal_dimension","Worst Fractal Dim.",0.055,0.208,0.084),
    ]
}

SAMPLE_MALIGNANT = {
    "mean_radius":17.99,"mean_texture":10.38,"mean_perimeter":122.8,"mean_area":1001.0,
    "mean_smoothness":0.1184,"mean_compactness":0.2776,"mean_concavity":0.3001,
    "mean_concave_points":0.1471,"mean_symmetry":0.2419,"mean_fractal_dimension":0.07871,
    "radius_error":1.095,"texture_error":0.9053,"perimeter_error":8.589,"area_error":153.4,
    "smoothness_error":0.006399,"compactness_error":0.04904,"concavity_error":0.05373,
    "concave_points_error":0.01587,"symmetry_error":0.03003,"fractal_dimension_error":0.006193,
    "worst_radius":25.38,"worst_texture":17.33,"worst_perimeter":184.6,"worst_area":2019.0,
    "worst_smoothness":0.1622,"worst_compactness":0.6656,"worst_concavity":0.7119,
    "worst_concave_points":0.2654,"worst_symmetry":0.4601,"worst_fractal_dimension":0.1189
}
SAMPLE_BENIGN = {
    "mean_radius":13.54,"mean_texture":14.36,"mean_perimeter":87.46,"mean_area":566.3,
    "mean_smoothness":0.09779,"mean_compactness":0.08129,"mean_concavity":0.06664,
    "mean_concave_points":0.04781,"mean_symmetry":0.1885,"mean_fractal_dimension":0.05766,
    "radius_error":0.2699,"texture_error":0.7886,"perimeter_error":2.058,"area_error":23.56,
    "smoothness_error":0.008462,"compactness_error":0.0146,"concavity_error":0.02387,
    "concave_points_error":0.01315,"symmetry_error":0.0198,"fractal_dimension_error":0.0023,
    "worst_radius":15.11,"worst_texture":19.26,"worst_perimeter":99.7,"worst_area":711.2,
    "worst_smoothness":0.144,"worst_compactness":0.1773,"worst_concavity":0.239,
    "worst_concave_points":0.1288,"worst_symmetry":0.2977,"worst_fractal_dimension":0.07259
}

# ── Session state ────────────────────────────────────────────────
if "defaults" not in st.session_state:
    st.session_state.defaults = SAMPLE_MALIGNANT.copy()
if "result" not in st.session_state:
    st.session_state.result = None

# ── Chart helpers ────────────────────────────────────────────────
def gauge(value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix":"%","font":{"size":26,"color":"#e2e8f0","family":"Space Grotesk"}},
        title={"text":"Model Confidence","font":{"size":12,"color":"#64748b","family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#1e293b",
                    "tickfont":{"color":"#475569","size":9}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"rgba(8,15,30,0.9)",
            "borderwidth":1,"bordercolor":"rgba(16,185,129,0.15)",
            "steps":[
                {"range":[0,33],"color":"rgba(16,185,129,0.06)"},
                {"range":[33,66],"color":"rgba(251,191,36,0.06)"},
                {"range":[66,100],"color":"rgba(239,68,68,0.06)"},
            ],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.85,"value":value}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=24,r=24,t=48,b=8),height=210,font={"family":"Inter"})
    return fig

def bar_chart(m_pct, b_pct):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Malignant","Benign"], y=[m_pct, b_pct],
        marker_color=["rgba(239,68,68,0.75)","rgba(16,185,129,0.75)"],
        marker_line_color=["#ef4444","#10b981"], marker_line_width=1.5,
        text=[f"{m_pct:.1f}%",f"{b_pct:.1f}%"],
        textposition="outside",
        textfont={"color":"#e2e8f0","size":14,"family":"Space Grotesk"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=16,b=10),height=230,
        yaxis=dict(range=[0,118],showgrid=False,showticklabels=False,zeroline=False),
        xaxis=dict(showgrid=False,tickfont={"color":"#94a3b8","size":13,"family":"Inter"}),
        showlegend=False, bargap=0.45,
    )
    return fig

def tree_chart(m_pct, b_pct):
    labels = ["Random Forest","Malignant","Benign"]
    parents = ["","Random Forest","Random Forest"]
    values  = [100, m_pct, b_pct]
    colors  = ["rgba(16,185,129,0.3)","rgba(239,68,68,0.7)","rgba(16,185,129,0.7)"]
    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        textfont=dict(family="Space Grotesk", size=13, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
        height=220,
    )
    return fig

# ── HERO ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🧬 Breast Cancer Detection AI</p>
    <p class="hero-sub">Advanced diagnostic assistant powered by Random Forest ensemble learning</p>
    <span class="hero-badge">🌲 RANDOM FOREST · 100 TREES · 30 FEATURES</span>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────
s1,s2,s3,s4,s5 = st.columns(5)
for col, val, lbl in zip(
    [s1,s2,s3,s4,s5],
    ["~96%","100","30","Benign","Malignant"],
    ["Accuracy","Trees","Features","Class 0","Class 1"]
):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Control Panel")
    st.markdown('<div class="sec-label">Quick Sample Load</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔴 Malignant", use_container_width=True):
            st.session_state.defaults = SAMPLE_MALIGNANT.copy()
            st.session_state.result = None
            st.rerun()
    with c2:
        if st.button("🟢 Benign", use_container_width=True):
            st.session_state.defaults = SAMPLE_BENIGN.copy()
            st.session_state.result = None
            st.rerun()

    st.markdown('<div class="sec-label">API Status</div>', unsafe_allow_html=True)
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"✅ Connected — {data.get('estimators',100)} trees")
        else:
            st.error("❌ API Error")
    except:
        st.warning("⚠️ API Offline\n\n`uvicorn main:app --reload`")

    st.markdown('<div class="sec-label">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
    <div class="tree-stat"><span class="tree-stat-label">Algorithm</span><span class="tree-stat-val">Random Forest</span></div>
    <div class="tree-stat"><span class="tree-stat-label">Estimators</span><span class="tree-stat-val">100 Trees</span></div>
    <div class="tree-stat"><span class="tree-stat-label">Features</span><span class="tree-stat-val">30</span></div>
    <div class="tree-stat"><span class="tree-stat-label">Dataset</span><span class="tree-stat-val">Wisconsin BC</span></div>
    <div class="tree-stat" style="border:none"><span class="tree-stat-label">Type</span><span class="tree-stat-val">Ensemble</span></div>
    </div>
    <div class="info-pill" style="margin-top:8px">
    ⚠️ For research & educational purposes only. Always consult a qualified medical professional for diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="sec-label">Patient Feature Input</div>', unsafe_allow_html=True)

    for grp_name, feats in FEATURES.items():
        with st.expander(grp_name, expanded=(grp_name == "🔵 Mean Values")):
            g1, g2 = st.columns(2)
            for i, (key, label, mn, mx, default) in enumerate(feats):
                with (g1 if i % 2 == 0 else g2):
                    st.number_input(
                        label, min_value=float(mn), max_value=float(mx),
                        value=float(st.session_state.defaults.get(key, default)),
                        format="%.5f", key=f"inp_{key}"
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Run Diagnosis", use_container_width=True)

with right:
    st.markdown('<div class="sec-label">Diagnostic Output</div>', unsafe_allow_html=True)

    if predict_btn:
        # ✅ Always read fresh widget values
        payload = {
            key: st.session_state[f"inp_{key}"]
            for grp in FEATURES.values()
            for key, *_ in grp
        }
        with st.spinner("Running Random Forest analysis..."):
            time.sleep(0.5)
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    st.session_state.result = response.json()
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Cannot connect to API\n\n**Run this first:**\n```\nuvicorn main:app --reload\n```")

    if st.session_state.result:
        r = st.session_state.result
        is_m = r["prediction"] == "Malignant"
        m_pct = r["malignant_probability"]
        b_pct = r["benign_probability"]

        # Result card
        if is_m:
            risk_bg = "rgba(239,68,68,0.15)"
            st.markdown(f"""
            <div class="card-malignant">
                <div class="card-icon">⚠️</div>
                <div class="card-title" style="color:#f87171">Malignant</div>
                <div class="card-sub">Cancerous tumor indicators detected</div>
                <span class="card-risk" style="background:{risk_bg};color:#fca5a5;border:1px solid rgba(239,68,68,0.3)">
                    🔴 {r['risk_level']}
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            risk_bg = "rgba(16,185,129,0.15)"
            st.markdown(f"""
            <div class="card-benign">
                <div class="card-icon">✅</div>
                <div class="card-title" style="color:#34d399">Benign</div>
                <div class="card-sub">No malignant indicators detected</div>
                <span class="card-risk" style="background:{risk_bg};color:#6ee7b7;border:1px solid rgba(16,185,129,0.3)">
                    🟢 {r['risk_level']}
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence gauge
        conf_color = "#ef4444" if is_m else "#10b981"
        st.plotly_chart(gauge(r["confidence"], conf_color),
                        use_container_width=True, config={"displayModeBar": False})

        # Two charts side by side
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<div class="sec-label">Probability</div>', unsafe_allow_html=True)
            st.plotly_chart(bar_chart(m_pct, b_pct),
                            use_container_width=True, config={"displayModeBar": False})
        with ch2:
            st.markdown('<div class="sec-label">Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(tree_chart(m_pct, b_pct),
                            use_container_width=True, config={"displayModeBar": False})

        # Prob cards
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(f'<div class="prob-card"><div class="prob-val" style="color:#f87171">{m_pct}%</div><div class="prob-lbl">Malignant</div></div>', unsafe_allow_html=True)
        with pc2:
            st.markdown(f'<div class="prob-card"><div class="prob-val" style="color:#34d399">{b_pct}%</div><div class="prob-lbl">Benign</div></div>', unsafe_allow_html=True)
        with pc3:
            st.markdown(f'<div class="prob-card"><div class="prob-val" style="color:#60a5fa">{r["confidence"]}%</div><div class="prob-lbl">Confidence</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if is_m:
            st.error("⚠️ Malignant indicators found. Please consult an oncologist immediately.")
        else:
            st.success("✅ Benign result. Continue with routine medical checkups.")

    else:
        st.markdown("""
        <div style="text-align:center;padding:70px 20px">
            <div style="font-size:3.5rem;margin-bottom:18px">🌲</div>
            <div style="font-size:1.05rem;color:#475569;font-weight:500">100 trees ready to analyze</div>
            <div style="font-size:0.9rem;color:#34d399;margin-top:8px;font-weight:600">Enter values → Click Run Diagnosis</div>
        </div>
        """, unsafe_allow_html=True)