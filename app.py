"""
ANN-Based Network Intrusion Detection System  |  Streamlit front end
Run:  streamlit run app.py

Expected files (created by your training notebook, see README_artifacts.md):
    artifacts/ann_model.keras      trained ANN
    artifacts/scaler.pkl           fitted scaler (optional but recommended)
    artifacts/classes.json         e.g. ["Normal", "Attack"]  (optional)
    artifacts/metrics.json         {"accuracy":..,"precision":..,"recall":..,"f1":..}
    artifacts/confusion_matrix.npy
    artifacts/classification_report.csv
    artifacts/history.csv          loss, val_loss, accuracy, val_accuracy
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG  - edit these to match YOUR project
# ----------------------------------------------------------------------------
PROJECT_NAME = "ANN-NIDS"
PROJECT_TITLE = "ANN-Based Network Intrusion Detection System"
ART = Path(__file__).parent / "artifacts"

TEAM = [  # replace with real names / roll numbers
    ("Shivendra Ghatage", "Roll No. 44", "ann model training & evaluation"),
    ("Sahil Khot", "Roll No. 45", "Front-end development"),
    ("Sai Chougale", "Roll No. 46", "data preprocessing "),
    ("Prajval Waghmode", "Roll No. 47", "ann model testing"),
    ("Amatualla bhori", "Roll No. 42", "data preprocessing"),
]
GUIDE = "Rajwardhan Todkar Sir"
DEPARTMENT = "Department of AI & ML, DY Patil Collage Of Agriculture & Techincal University,Talsande"

# (column name, label, min, max (None = no limit), help text)
# IMPORTANT: keep the ORDER identical to the columns used when you trained the ANN.
FEATURES = [
    ("dur", "Duration (dur)", 0.0, None, "Record total duration in seconds"),
    ("spkts", "Source packets (spkts)", 0.0, None, "Packets from source to destination"),
    ("dpkts", "Destination packets (dpkts)", 0.0, None, "Packets from destination to source"),
    ("sbytes", "Source bytes (sbytes)", 0.0, None, "Bytes from source to destination"),
    ("dbytes", "Destination bytes (dbytes)", 0.0, None, "Bytes from destination to source"),
    ("rate", "Rate", 0.0, None, "Packets per second"),
    ("sttl", "Source TTL (sttl)", 0.0, 255.0, "0 - 255"),
    ("dttl", "Destination TTL (dttl)", 0.0, 255.0, "0 - 255"),
    ("sload", "Source load (sload)", 0.0, None, "Source bits per second"),
    ("dload", "Destination load (dload)", 0.0, None, "Destination bits per second"),
    ("sinpkt", "Source inter-packet time (sinpkt)", 0.0, None, "ms"),
    ("dinpkt", "Destination inter-packet time (dinpkt)", 0.0, None, "ms"),
    ("sjit", "Source jitter (sjit)", 0.0, None, "ms"),
    ("djit", "Destination jitter (djit)", 0.0, None, "ms"),
    ("tcprtt", "TCP round-trip time (tcprtt)", 0.0, None, "seconds"),
]

PAGES = {
    "home": "Home",
    "prediction": "Intrusion Prediction",
    "performance": "Model Performance",
    "methodology": "Methodology",
    "about": "About & Team",
}

# ----------------------------------------------------------------------------
# PAGE SETUP + THEME (matches the Site24x7 look: white nav, dark gradient hero,
# mint headline, white form card, orange call-to-action)
# ----------------------------------------------------------------------------
st.set_page_config(page_title=PROJECT_TITLE, page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --mint:#aaffd9; --ink:#ffffff; --muted:#c9d1cd; --bg:#000000;
  --pill:#0b0b0b; --pill-border:#2b2b2b; --orange:#f5731e; --orange-d:#dc6314;
  --card:#ffffff; --field:#ececec; --field-border:#b9b9b9; --text-dark:#1f1f1f; --link:#0b6bcb;
}
html, body, [class*="css"], .stApp, button, input, textarea{
  font-family:'Inter','Segoe UI',Arial,sans-serif !important; letter-spacing:-0.01em;
}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stToolbar"]{display:none !important;}
.stApp{background:linear-gradient(180deg,#2d4f46 0px,#0c1512 230px,#000 440px) no-repeat, #000;color:var(--ink);}
.block-container{max-width:1180px; padding:104px 24px 60px 24px !important;}

/* ---------- top navigation bar ---------- */
.topnav{position:fixed; top:0; left:0; right:0; height:64px; background:#fff; z-index:9999;
  display:flex; align-items:center; justify-content:space-between; padding:0 7vw;
  border-bottom:1px solid #e6e6e6;}
.brand{display:flex; align-items:center; gap:10px; text-decoration:none !important;}
.brand .t1{font-size:11px; font-weight:600; color:#333; line-height:1.1;}
.brand .t2{font-size:17px; font-weight:600; color:#111; line-height:1.15;}
.navlinks{display:flex; gap:30px; align-items:center;}
.navlinks a{color:#1b1b1b !important; font-size:15px; font-weight:500; text-decoration:none !important;
  padding:21px 0 19px; border-bottom:3px solid transparent;}
.navlinks a:hover{color:#0a8f5b !important;}
.navlinks a.active{border-bottom-color:#17b978; color:#0a7a50 !important;}
@media (max-width:900px){.navlinks{gap:14px}.navlinks a{font-size:12px}.brand .t1{display:none}}

/* ---------- typography ---------- */
h1.hero{font-size:68px; line-height:1.04; font-weight:600; color:var(--mint); margin:34px 0 28px; letter-spacing:-0.03em;}
h1.hero.small{font-size:52px; margin:22px 0 10px;}
p.sub{font-size:30px; line-height:1.35; font-weight:500; color:var(--ink); letter-spacing:-0.025em; max-width:640px;}
p.sub.small{font-size:20px; font-weight:400; color:var(--muted);}
h2.sec{font-size:30px; font-weight:600; color:var(--mint); margin:46px 0 14px;}
p.body, li.body{font-size:17px; line-height:1.65; color:var(--muted);}
.pill{display:flex; align-items:center; gap:14px; width:fit-content; background:var(--pill);
  border:1px solid var(--pill-border); border-radius:14px; padding:13px 28px 13px 18px;
  margin:14px 0; font-size:17px; color:#f1f1f1;}
.pill svg{flex:0 0 auto}

/* ---------- white form card ---------- */
div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card) !important; border:none !important; border-radius:12px !important; padding:14px 18px 10px !important;}
div[data-testid="stForm"] *, div[data-testid="stVerticalBlockBorderWrapper"] label p{color:var(--text-dark);}
.card-title{font-size:26px; font-weight:600; color:#ffffff !important; margin:6px 0 14px;}
.card-text{font-size:15px; color:#ffffff !important; line-height:1.55;}
[data-baseweb="input"], [data-baseweb="base-input"]{background:var(--field) !important; border-radius:3px !important;}
[data-baseweb="input"]{border:1px solid var(--field-border) !important;}
[data-baseweb="input"] input{background:var(--field) !important; color:#222 !important; -webkit-text-fill-color:#222 !important;}
div[data-testid="stNumberInput"] button{background:var(--field) !important; color:#444 !important;}
label[data-testid="stWidgetLabel"] p{font-size:13.5px; font-weight:500;}

/* ---------- orange call-to-action ---------- */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button, a.cta{
  background:var(--orange) !important; color:#fff !important; border:none !important; border-radius:3px !important;
  width:100%; padding:13px 10px !important; font-weight:700 !important; font-size:16px !important;
  letter-spacing:0.01em; text-transform:uppercase; transition:background .15s;}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover, a.cta:hover{background:var(--orange-d) !important;}
div.stButton > button p, div[data-testid="stFormSubmitButton"] > button p{color:#fff !important; font-weight:700;}
a.cta{display:block; text-align:center; text-decoration:none !important; margin-top:14px;}
.cta-sub{display:block; font-size:11px; font-weight:600; opacity:.95;}

/* ---------- result + metric blocks ---------- */
.result{border-radius:12px; padding:26px 30px; margin-top:22px; border:1px solid;}
.result.ok{background:#07231a; border-color:#17b978;} .result.bad{background:#2a0b0b; border-color:#ff5a5a;}
.result .big{font-size:34px; font-weight:700;}
.result.ok .big{color:var(--mint);} .result.bad .big{color:#ff8d8d;}
.result .meta{font-size:16px; color:var(--muted); margin-top:6px;}
.metric{background:var(--pill); border:1px solid var(--pill-border); border-radius:14px; padding:20px 22px;}
.metric .v{font-size:36px; font-weight:600; color:var(--mint);} .metric .k{font-size:14px; color:var(--muted);}
.step{background:var(--pill); border:1px solid var(--pill-border); border-radius:14px; padding:16px 18px; height:100%;}
.step b{color:var(--mint); font-size:16px;} .step span{display:block; color:var(--muted); font-size:14px; margin-top:4px;}
.person{background:var(--pill); border:1px solid var(--pill-border); border-radius:14px; padding:18px 22px; margin-bottom:12px;}
.person b{font-size:18px; color:#fff;} .person span{color:var(--muted); font-size:14px; display:block;}
.stDataFrame, [data-testid="stTable"]{border-radius:10px;}
a{color:#7fd6ff;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CHECK = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#bfc7c3" stroke-width="1.8" '
         'stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-5.2-8.2"/>'
         '<path d="M8 12l3 3 7-8"/></svg>')
LOGO = ('<svg width="40" height="34" viewBox="0 0 48 40" fill="none" stroke="#17b978" stroke-width="3.4" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M10 32c-5 0-8-3-8-7s3-7 8-7c1-6 6-10 12-10 '
        '7 0 12 4 12 10 5 0 8 3 8 7s-3 7-8 7z"/><path d="M24 12l8 3v6c0 5-4 8-8 10-4-2-8-5-8-10v-6z" '
        'stroke-width="2.6"/></svg>')


def navbar(active: str) -> None:
    links = "".join(
        f'<a class="{"active" if k == active else ""}" href="?page={k}" target="_self">{v}</a>'
        for k, v in PAGES.items())
    st.markdown(
        f'<div class="topnav"><a class="brand" href="?page=home" target="_self">{LOGO}'
        f'<div><div class="t1">B.Tech AI &amp; ML</div><div class="t2">{PROJECT_NAME}</div></div></a>'
        f'<div class="navlinks">{links}</div></div>', unsafe_allow_html=True)


def pill(text: str) -> str:
    return f'<div class="pill">{CHECK}<span>{text}</span></div>'


# ----------------------------------------------------------------------------
# Loading real artifacts (no fake numbers anywhere)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    path = next((p for p in (ART / "ann_model.keras", ART / "ann_model.h5") if p.exists()), None)
    if path is None:
        return None, None, None, "Model file not found in the 'artifacts' folder."
    try:
        from tensorflow import keras  # imported lazily so the UI still opens without TF
        import joblib
        model = keras.models.load_model(path)
        scaler = joblib.load(ART / "scaler.pkl") if (ART / "scaler.pkl").exists() else None
        classes = (json.loads((ART / "classes.json").read_text())
                   if (ART / "classes.json").exists() else ["Normal", "Attack"])
        return model, scaler, classes, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"Could not load the model: {exc}"


def load_classes():
    f = ART / "classes.json"
    return json.loads(f.read_text()) if f.exists() else ["Normal", "Attack"]


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
def page_home():
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<h1 class="hero">Network<br>intrusion detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub">Classify network traffic as normal or malicious using an '
                    'Artificial Neural Network trained on the UNSW-NB15 dataset.</p>', unsafe_allow_html=True)
        st.markdown(pill("Detect attacks from traffic flow features.")
                    + pill("Trained and evaluated on UNSW-NB15.")
                    + pill("Review accuracy, precision, recall and F1-score."), unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">TRY THE DETECTOR</div>'
                        '<p class="card-text">Enter the traffic features of one network flow and the ' 
                        'ANN returns its prediction. All results come from the trained model; '
                        'nothing is simulated.</p>'
                        '<a class="cta" href="?page=prediction" target="_self">Start prediction'
                        '<span class="cta-sub">Uses the trained ANN model</span></a>'
                        '<a class="cta" style="background:#1f1f1f !important" href="?page=performance" '
                        'target="_self">View model performance</a><div style="height:10px"></div>',
                        unsafe_allow_html=True)

    st.markdown('<h2 class="sec">What is a Network Intrusion Detection System?</h2>', unsafe_allow_html=True)
    st.markdown('<p class="body">A Network Intrusion Detection System (NIDS) inspects network traffic and '
                'flags activity that looks like an attack, such as denial-of-service, reconnaissance or '
                'exploits. Instead of fixed rules, this project trains an ANN to learn the difference '
                'between normal and malicious flows from labelled examples.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>Dataset</b><span>UNSW-NB15, created by the Australian Centre for '
                'Cyber Security (ACCS).</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>Model</b><span>Feed-forward Artificial Neural Network '
                '(fully connected layers).</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>Output</b><span>Normal traffic or attack detected'
                '</span></div>', unsafe_allow_html=True)


def page_prediction():
    st.markdown('<h1 class="hero small">Intrusion prediction</h1>'
                '<p class="sub small">Enter the flow features below and select Predict.</p>',
                unsafe_allow_html=True)
    model, scaler, classes, err = load_model()
    if err:
        st.warning(f"{err} Train the ANN and save it into the `artifacts` folder, then reload this page.")

    with st.form("predict_form"):
        st.markdown('<div class="card-title">Network traffic features</div>', unsafe_allow_html=True)
        values = {}
        cols = st.columns(3)
        for i, (name, label, lo, hi, helptext) in enumerate(FEATURES):
            with cols[i % 3]:
                values[name] = st.number_input(label, min_value=lo, max_value=hi, value=0.0,
                                               format="%.6f", help=helptext, key=f"f_{name}")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return
    if model is None:
        st.error("Prediction is unavailable because the trained model is not loaded.")
        return

    x = np.array([[values[f[0]] for f in FEATURES]], dtype=float)
    problems = []
    if not np.isfinite(x).all():
        problems.append("All inputs must be valid numbers.")
    if (x < 0).any():
        problems.append("Negative values are not allowed.")
    for name, label, lo, hi, _ in FEATURES:
        if hi is not None and values[name] > hi:
            problems.append(f"{label} must be at most {hi:g}.")
    if not (x > 0).any():
        problems.append("Every value is zero. Enter the flow's actual feature values.")
    if problems:
        for p in problems:
            st.error(p)
        return

    xs = scaler.transform(x) if scaler is not None else x
    try:
        out = model.predict(xs, verbose=0)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The model rejected this input. Check that FEATURES matches the training columns. ({exc})")
        return

    if out.shape[-1] == 1:  # sigmoid output
        p_attack = float(out[0][0])
        is_attack = p_attack >= 0.5
        label = "Attack" if is_attack else "Normal"
        conf = p_attack if is_attack else 1 - p_attack
    else:  # softmax output (multiclass)
        idx = int(np.argmax(out[0]))
        label = classes[idx] if idx < len(classes) else f"Class {idx}"
        is_attack = label.strip().lower() != "normal"
        conf = float(out[0][idx])

    if is_attack:
        title = "Attack detected" if label.lower() == "attack" else f"Attack detected: {label}"
        st.markdown(f'<div class="result bad"><div class="big">{title}</div>'
                    f'<div class="meta">Model confidence: {conf*100:.2f}%</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result ok"><div class="big">Normal traffic</div>'
                    f'<div class="meta">Model confidence: {conf*100:.2f}%</div></div>', unsafe_allow_html=True)
    st.caption("This is a prediction for the submitted feature values only. It is not live network monitoring.")


def metric_card(k, v):
    return f'<div class="metric"><div class="v">{v}</div><div class="k">{k}</div></div>'


def page_performance():
    st.markdown('<h1 class="hero small">Model performance</h1>'
                '<p class="sub small">Results measured on the held-out test set.</p>', unsafe_allow_html=True)
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
                         "ytick.color": "white", "axes.edgecolor": "#555"})

    mf = ART / "metrics.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        cols = st.columns(4)
        for col, (k, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                         ("Recall", "recall"), ("F1-score", "f1")]):
            val = m.get(key)
            col.markdown(metric_card(k, f"{val*100:.2f}%" if isinstance(val, (int, float)) else "n/a"),
                         unsafe_allow_html=True)
    else:
        st.info("`artifacts/metrics.json` not found. Save the test-set metrics from your notebook to show them here.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<h2 class="sec" style="margin-top:30px">Confusion matrix</h2>', unsafe_allow_html=True)
        cm_f = ART / "confusion_matrix.npy"
        if cm_f.exists():
            cm = np.load(cm_f)
            names = load_classes()
            names = names if len(names) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=names, yticklabels=names,
                        cbar=False, ax=ax, annot_kws={"color": "black"})
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig, transparent=True)
        else:
            st.info("`artifacts/confusion_matrix.npy` not found.")
    with c2:
        st.markdown('<h2 class="sec" style="margin-top:30px">Training history</h2>', unsafe_allow_html=True)
        hf = ART / "history.csv"
        if hf.exists():
            h = pd.read_csv(hf)
            fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True)
            fig.patch.set_alpha(0)
            for ax, (a, b, t) in zip(axes, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]):
                ax.set_facecolor("none")
                if a in h: ax.plot(h[a], color="#aaffd9", label="train")
                if b in h: ax.plot(h[b], color="#f5731e", label="validation")
                ax.set_ylabel(t); ax.legend(frameon=False, labelcolor="white")
            axes[-1].set_xlabel("Epoch")
            fig.tight_layout()
            st.pyplot(fig, transparent=True)
        else:
            st.info("`artifacts/history.csv` not found.")

    rf = ART / "classification_report.csv"
    if rf.exists():
        st.markdown('<h2 class="sec">Classification report</h2>', unsafe_allow_html=True)
        st.dataframe(pd.read_csv(rf, index_col=0), use_container_width=True)


def page_methodology():
    st.markdown('<h1 class="hero small">Methodology</h1>'
                '<p class="sub small">From raw UNSW-NB15 records to a trained ANN.</p>', unsafe_allow_html=True)
    steps = [("Data collection", "UNSW-NB15 training and testing CSV files."),
             ("Preprocessing", "Remove duplicates, handle missing values, encode categorical fields."),
             ("Feature scaling", "Scale numeric features (e.g. StandardScaler / MinMaxScaler)."),
             ("ANN training", "Dense layers with ReLU, dropout, sigmoid or softmax output."),
             ("Evaluation", "Accuracy, precision, recall, F1-score, confusion matrix."),
             ("Deployment", "Saved model and scaler loaded in this Streamlit app.")]
    cols = st.columns(3)
    for i, (t, d) in enumerate(steps):
        cols[i % 3].markdown(f'<div class="step" style="margin-bottom:14px"><b>{t}</b><span>{d}</span></div>',
                             unsafe_allow_html=True)
    st.markdown('<h2 class="sec">Problem statement</h2><p class="body">Traditional signature-based '
                'intrusion detection cannot keep up with new and changing attacks. This project builds an '
                'ANN that learns patterns in network flow features to separate normal traffic from attacks.</p>'
                '<h2 class="sec">Objectives</h2>'
                '<ul><li class="body">Preprocess the UNSW-NB15 dataset.</li>'
                '<li class="body">Design, train and tune an ANN classifier.</li>'
                '<li class="body">Evaluate it with standard classification metrics.</li>'
                '<li class="body">Present the model through a simple web interface.</li></ul>'
                '<h2 class="sec">How the ANN works</h2><p class="body">Each input feature feeds a layer of '
                'neurons. Every neuron computes a weighted sum, applies an activation function, and passes the '
                'result forward. During training, backpropagation adjusts the weights to reduce the loss. '
                'The final layer outputs the probability of attack (or one probability per attack class).</p>'
                '<h2 class="sec">Technologies used</h2><p class="body">Python, NumPy, Pandas, Matplotlib, '
                'Seaborn, SciPy, scikit-learn, TensorFlow / Keras, Streamlit.</p>', unsafe_allow_html=True)


def page_about():
    st.markdown('<h1 class="hero small">About &amp; team</h1>'
                f'<p class="sub small">{PROJECT_TITLE}</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec" style="margin-top:20px">Team members</h2>', unsafe_allow_html=True)
    for n, r, role in TEAM:
        st.markdown(f'<div class="person"><b>{n}</b><span>{r} &nbsp;|&nbsp; {role}</span></div>',
                    unsafe_allow_html=True)
    st.markdown(f'<h2 class="sec">Guide and department</h2><p class="body">{GUIDE}<br>{DEPARTMENT}</p>'
                '<h2 class="sec">Dataset and references</h2>'
                '<ul><li class="body">Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for '
                'network intrusion detection systems. Military Communications and Information Systems '
                'Conference (MilCIS), IEEE.</li>'
                '<li class="body">Moustafa, N. and Slay, J. (2016). The evaluation of network anomaly detection '
                'systems: statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data '
                'set. Information Security Journal: A Global Perspective, 25(1-3), 18-31.</li>'
                '<li class="body">Dataset page: UNSW Canberra Cyber, UNSW-NB15 dataset.</li></ul>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
page = st.query_params.get("page", "home")
if page not in PAGES:
    page = "home"
navbar(page)
{"home": page_home, "prediction": page_prediction, "performance": page_performance,
 "methodology": page_methodology, "about": page_about}[page]()
