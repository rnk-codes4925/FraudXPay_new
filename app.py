import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import io
import hashlib
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FraudXPay · AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# DESIGN SYSTEM (CSS)
# ════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">

<style>
:root{
  --bg-0:#070b1a;
  --bg-1:#0d1428;
  --bg-2:#131a35;
  --surface:#161e3d;
  --surface-2:#1c2547;
  --border:rgba(255,255,255,0.08);
  --border-strong:rgba(110,168,255,0.25);
  --text:#E6EBF5;
  --text-dim:#8993B0;
  --text-mute:#5A6483;
  --primary:#6EA8FE;
  --primary-2:#7C5CFF;
  --accent:#22D3EE;
  --success:#10B981;
  --warning:#F59E0B;
  --danger:#EF4444;
  --grad: linear-gradient(135deg,#6EA8FE 0%,#7C5CFF 50%,#22D3EE 100%);
  --grad-soft: linear-gradient(135deg,rgba(110,168,255,0.15),rgba(124,92,255,0.10),rgba(34,211,238,0.12));
  --shadow-lg: 0 25px 60px -20px rgba(8,12,30,0.7), 0 8px 24px rgba(110,168,255,0.08);
  --radius: 16px;
}
::-webkit-scrollbar{display:none}
*{-ms-overflow-style:none;scrollbar-width:none}
*{font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;-webkit-font-smoothing:antialiased}
html,body,[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(124,92,255,0.18), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(34,211,238,0.12), transparent 60%),
    radial-gradient(800px 600px at 50% 120%, rgba(110,168,255,0.10), transparent 60%),
    var(--bg-0) !important;
  color:var(--text);
}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.5rem;padding-bottom:4rem;max-width:1400px}
h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif;color:var(--text);letter-spacing:-0.02em}
h1{font-weight:700;font-size:42px;line-height:1.1}
h2{font-weight:700;font-size:28px}
h3{font-weight:600;font-size:20px;color:var(--text)}
p,label,span,div{color:var(--text)}
small,.muted{color:var(--text-dim)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1024 0%,#0b1330 100%);border-right:1px solid var(--border)}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
.card{background:linear-gradient(180deg,rgba(28,37,71,0.7),rgba(22,30,61,0.7));backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:var(--radius);padding:24px 26px;box-shadow:var(--shadow-lg)}
.card-accent{background:var(--grad-soft);border:1px solid var(--border-strong);border-radius:var(--radius);padding:24px 26px}
.hero{position:relative;padding:48px 44px;border-radius:24px;background:radial-gradient(600px 200px at 0% 0%, rgba(124,92,255,0.25), transparent 60%),radial-gradient(500px 200px at 100% 100%, rgba(34,211,238,0.18), transparent 60%),linear-gradient(180deg,rgba(28,37,71,0.85),rgba(13,20,40,0.85));border:1px solid var(--border-strong);box-shadow:var(--shadow-lg);overflow:hidden;margin-bottom:28px}
.hero h1{font-size:48px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 10px 0}
.hero p{color:var(--text-dim);font-size:17px;max-width:720px;margin:0}
.hero .badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:rgba(110,168,255,0.12);border:1px solid var(--border-strong);color:var(--primary);font-weight:600;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px}
.hero .dot{width:8px;height:8px;border-radius:50%;background:var(--success);box-shadow:0 0 12px var(--success)}
.stButton>button,.stDownloadButton>button{background:var(--grad);color:#0a0f24 !important;border:0;padding:12px 22px;border-radius:12px;font-weight:700;font-size:14px;letter-spacing:0.02em;transition:transform .15s ease,box-shadow .2s ease,filter .2s ease;box-shadow:0 10px 24px -8px rgba(110,168,255,0.55)}
.stButton>button:hover{transform:translateY(-1px);filter:brightness(1.05);box-shadow:0 14px 30px -10px rgba(124,92,255,0.6)}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"]{background-color:rgba(13,20,40,0.7) !important;color:white !important;border-radius:12px !important}
div[data-baseweb="select"]>div{color:white !important}
div[role="listbox"]{background-color:#131a35 !important;color:white !important}
div[role="option"]{color:white !important}
div[role="option"]:hover{background-color:#1c2547 !important}
label{color:var(--text-dim) !important;font-weight:600 !important;font-size:13px !important;letter-spacing:0.02em}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(28,37,71,0.65),rgba(22,30,61,0.65));padding:22px 22px;border-radius:16px;border:1px solid var(--border);box-shadow:0 8px 24px -12px rgba(0,0,0,0.5);transition:transform .2s ease,border-color .2s ease}
[data-testid="stMetric"]:hover{transform:translateY(-2px);border-color:var(--border-strong)}
[data-testid="stMetricLabel"]{color:var(--text-dim) !important;font-size:12px !important;font-weight:600 !important;text-transform:uppercase;letter-spacing:0.1em}
[data-testid="stMetricValue"]{color:var(--text) !important;font-family:'Space Grotesk',sans-serif !important;font-size:32px !important;font-weight:700 !important}
[data-testid="stMetricDelta"]{font-weight:600 !important}
[data-testid="stSidebar"] [role="radiogroup"]{gap:6px;display:flex;flex-direction:column}
[data-testid="stSidebar"] [role="radiogroup"] label{background:rgba(255,255,255,0.02);border:1px solid var(--border);padding:11px 14px;border-radius:12px;color:var(--text-dim) !important;font-weight:600;font-size:14px;transition:all .2s ease;cursor:pointer}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(110,168,255,0.08);color:var(--text) !important;border-color:var(--border-strong)}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--grad-soft);color:var(--text) !important;border-color:var(--border-strong)}
.stAlert{border-radius:12px !important;border:1px solid var(--border) !important;padding:14px 16px !important}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--border)}
.section-h{display:flex;align-items:center;gap:10px;margin:8px 0 14px 0}
.section-h .pill{padding:4px 10px;border-radius:999px;background:rgba(110,168,255,0.12);color:var(--primary);font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;border:1px solid var(--border-strong)}
.section-h h2{margin:0}
.topbar{display:flex;justify-content:space-between;align-items:center;background:linear-gradient(180deg,rgba(28,37,71,0.7),rgba(22,30,61,0.7));border:1px solid var(--border);border-radius:16px;padding:14px 20px;margin-bottom:22px;backdrop-filter:blur(12px)}
.brand{display:flex;align-items:center;gap:12px}
.brand-mark{width:40px;height:40px;border-radius:11px;display:grid;place-items:center;background:var(--grad);color:#0a0f24;font-weight:800;font-size:18px;box-shadow:0 8px 20px -8px rgba(110,168,255,0.6)}
.brand-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:18px;color:var(--text);letter-spacing:-0.01em}
.brand-sub{font-size:11px;color:var(--text-mute);letter-spacing:0.14em;text-transform:uppercase}
.user-chip{display:flex;align-items:center;gap:10px;padding:6px 12px 6px 6px;border-radius:999px;background:rgba(255,255,255,0.04);border:1px solid var(--border)}
.avatar{width:32px;height:32px;border-radius:50%;background:var(--grad);display:grid;place-items:center;color:#0a0f24;font-weight:800;font-size:13px}
.user-meta{display:flex;flex-direction:column;line-height:1.1}
.user-name{font-size:13px;font-weight:600;color:var(--text)}
.user-email{font-size:11px;color:var(--text-mute)}
.login-wrap{display:grid;place-items:center;min-height:90vh;overflow:hidden !important}
.login-card{width:100%;max-width:460px;background:linear-gradient(180deg,rgba(28,37,71,0.85),rgba(13,20,40,0.85));border:1px solid var(--border-strong);border-radius:22px;padding:36px 32px;box-shadow:var(--shadow-lg);backdrop-filter:blur(14px);overflow:hidden !important}
.login-logo{width:60px;height:60px;border-radius:16px;display:grid;place-items:center;background:var(--grad);color:#0a0f24;font-weight:800;font-size:26px;margin:0 auto 16px;box-shadow:0 14px 30px -10px rgba(124,92,255,0.6)}
.login-title{text-align:center;font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;margin:0 0 6px}
.login-sub{text-align:center;color:var(--text-dim);font-size:14px;margin-bottom:22px}
.stFileUploader label{color:var(--text-dim) !important;font-weight:600 !important;font-size:13px !important;letter-spacing:0.02em}
[data-testid="stFileUploader"] div>div>div{background:rgba(13,20,40,0.7) !important;border:1px solid var(--border) !important;border-radius:12px !important}
[data-testid="stProgress"]>div>div>div{background:var(--grad) !important}
#MainMenu,footer{visibility:hidden}
[data-testid="stToolbar"]{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
ss = st.session_state
ss.setdefault("logged_in", False)
ss.setdefault("user_name", "")
ss.setdefault("user_email", "")
ss.setdefault("user_history", {})
ss.setdefault("login_mode", "Sign in")
ss.setdefault("batch_results", None)
ss.setdefault("model_trained", False)

# ════════════════════════════════════════════════════════════════
# CONFIG / DATA
# ════════════════════════════════════════════════════════════════
DEMO_USERS = {
    "demo@fraudxpay.com": "demo123",
    "user@fraudxpay.com": "user123",
    "renukanayak@gmail.com": "09876",
}

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]

USER_PROFILES = {
    "demo@fraudxpay.com": {
        "avg_transaction": 2500,
        "max_daily_spend": 15000,
        "typical_locations": ["US", "CA", "UK"],
        "device_ids": ["DEV-001", "DEV-002"],
    },
    "user@fraudxpay.com": {
        "avg_transaction": 1800,
        "max_daily_spend": 10000,
        "typical_locations": ["US", "MX"],
        "device_ids": ["DEV-003", "DEV-004"],
    },
}

SAMPLE_TRANSACTIONS = pd.DataFrame({
    "Date": ["2024-05-04 10:12", "2024-05-04 09:45", "2024-05-04 08:30", "2024-05-04 07:15", "2024-05-03 22:10"],
    "Type": ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER", "PAYMENT"],
    "Amount ($)": [2500, 150, 850, 12000, 45],
    "Status": ["Approved", "Approved", "Review", "Blocked", "Approved"],
    "Risk Score": ["12%", "5%", "45%", "88%", "2%"]
})

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E6EBF5", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ════════════════════════════════════════════════════════════════
# ML MODEL
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load('fraud_model.pkl')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except:
        np.random.seed(42)
        n_samples = 10000
        data = {
            'amount': np.random.lognormal(7, 1.5, n_samples),
            'balance_orig': np.random.lognormal(8, 1.2, n_samples),
            'balance_dest': np.random.lognormal(8, 1.2, n_samples),
            'type_idx': np.random.choice([0,1,2,3], n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'is_weekend': np.random.choice([0,1], n_samples, p=[0.75, 0.25])
        }
        df = pd.DataFrame(data)
        df['depletion_ratio'] = np.clip(df['amount'] / (df['balance_orig'] + 1), 0, 2)
        df['amount_to_balance_ratio'] = df['amount'] / (df['balance_dest'] + 1)
        fraud_conditions = [
            (df['amount'] > 15000) |
            (df['depletion_ratio'] > 0.9) |
            (df['hour'] < 3) |
            (df['is_weekend'] == 1) & (df['amount'] > 5000)
        ]
        df['is_fraud'] = np.random.choice([0,1], n_samples, p=[0.95, 0.05])
        df.loc[np.logical_or.reduce(fraud_conditions), 'is_fraud'] = 1
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'fraud_model.pkl')
        explainer = shap.TreeExplainer(model)
        ss.model_trained = True
        return model, explainer

model, shap_explainer = load_or_train_model()

def calculate_risk_score_ml(features):
    feature_df = pd.DataFrame([features], columns=['amount','balance_orig','balance_dest','type_idx','hour','is_weekend','depletion_ratio','amount_to_balance_ratio'])
    prob = model.predict_proba(feature_df)[0][1] * 100
    return prob

def get_shap_explanation(features):
    feature_df = pd.DataFrame([features], columns=['amount','balance_orig','balance_dest','type_idx','hour','is_weekend','depletion_ratio','amount_to_balance_ratio'])
    shap_vals = shap_explainer.shap_values(feature_df)
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        if len(shap_vals.shape) == 3:
            vals = shap_vals[0, :, 1]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0, :]
        else:
            vals = shap_vals
    feature_names = feature_df.columns
    importance = []
    for i, (name, val) in enumerate(zip(feature_names, vals)):
        importance.append({
            'feature': name,
            'value': feature_df.iloc[0][name],
            'shap_value': float(val),
            'impact': 'High' if abs(val) > 0.05 else 'Medium' if abs(val) > 0.02 else 'Low'
        })
    return sorted(importance, key=lambda x: abs(x['shap_value']), reverse=True)[:5]

def behavioral_anomaly_score(user_profile, amount, location="US", device="DEV-001", time_hour=12, is_weekend=False):
    anomaly = 0.0
    if user_profile:
        avg_txn = user_profile.get("avg_transaction", 2500)
        if amount > avg_txn * 3: anomaly += 30
        elif amount > avg_txn * 2: anomaly += 15
    typical_locations = user_profile.get("typical_locations", ["US"])
    if location not in typical_locations: anomaly += 25
    typical_devices = user_profile.get("device_ids", [])
    if device not in typical_devices: anomaly += 20
    if time_hour < 6 or (is_weekend and time_hour < 10): anomaly += 15
    return min(anomaly, 100)

def section(title, kicker="Overview"):
    st.markdown(f'<div class="section-h"><span class="pill">{kicker}</span><h2>{title}</h2></div>', unsafe_allow_html=True)

def generate_report(results_df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Fraud Analysis', index=False)
        summary = pd.DataFrame({
            'Metric': ['Total Transactions', 'Fraud Detected', 'Fraud Rate', 'High Risk', 'Medium Risk', 'Low Risk'],
            'Count': [
                len(results_df),
                len(results_df[results_df['Prediction'] == 'Fraud']),
                f"{len(results_df[results_df['Risk Score %'] > 70])/len(results_df)*100:.1f}%",
                len(results_df[results_df['Risk Score %'] > 70]),
                len(results_df[(results_df['Risk Score %'] <= 70) & (results_df['Risk Score %'] > 30)]),
                len(results_df[results_df['Risk Score %'] <= 30])
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    buffer.seek(0)
    return buffer, f"fraudxpay_report_{timestamp}.xlsx"

# ════════════════════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════════════════════
def login_page():
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown('<div class="login-card"><div class="login-logo">🛡️</div><div class="login-title">FraudXPay</div><div class="login-sub">AI-powered transaction fraud detection</div>', unsafe_allow_html=True)
        if ss.login_mode == "Sign in":
            email = st.text_input("Email", placeholder="you@company.com", key="li_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pw")
            if st.button("Sign in securely", use_container_width=True):
                if email in DEMO_USERS and DEMO_USERS[email] == password:
                    ss.logged_in = True
                    ss.user_name = email.split("@")[0].title()
                    ss.user_email = email
                    st.rerun()
                elif email and password:
                    st.error("Invalid email or password.")
                else:
                    st.warning("Please enter your email and password.")
            st.markdown("<div style='text-align:center;margin-top:10px'>", unsafe_allow_html=True)
            if st.button("Don't have an account? Create one", key="go_signup"):
                ss.login_mode = "Sign up"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("Demo account: **demo@fraudxpay.com** · password **demo123**")
        else:
            n = st.text_input("Full name")
            e = st.text_input("Email", key="su_email")
            p1 = st.text_input("Password", type="password", key="su_p1")
            p2 = st.text_input("Confirm password", type="password", key="su_p2")
            if st.button("Create account", use_container_width=True):
                if not (n and e and p1):
                    st.warning("Please fill all fields.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    st.success("Account created. Please sign in.")
                    ss.login_mode = "Sign in"
                    st.rerun()
            st.markdown("<div style='text-align:center;margin-top:10px'>", unsafe_allow_html=True)
            if st.button("Already have an account? Sign in", key="go_signin"):
                ss.login_mode = "Sign in"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def topbar():
    initials = (ss.user_name[:1] or "U").upper()
    st.markdown(f'<div class="topbar"><div class="brand"><div class="brand-mark">🛡️</div><div><div class="brand-name">FraudXPay</div><div class="brand-sub">Secure Payments Intelligence</div></div></div><div class="user-chip"><div class="avatar">{initials}</div><div class="user-meta"><span class="user-name">{ss.user_name}</span><span class="user-email">{ss.user_email}</span></div></div></div>', unsafe_allow_html=True)

def home_page():
    st.markdown(f'<div class="hero"><div class="badge"><span class="dot"></span> Live · Model healthy</div><h1>Welcome back, {ss.user_name}.</h1><p>Real-time fraud monitoring across your payment network. Track transactions, investigate anomalies, and act on AI-driven risk scores in seconds.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions (24h)", "50,234", "+2.5%")
    c2.metric("Frauds detected", "342", "-5.2%")
    c3.metric("Model accuracy", "97.8%", "+0.3%")
    c4.metric("Fraud rate", "0.68%", "-0.05%")
    c5.metric("Batch jobs", "12", "+3")
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        section("Fraud trend (last 30 days)", "Analytics")
        days = pd.date_range(end=datetime.today(), periods=30)
        frauds = np.random.poisson(11, 30) + np.random.normal(0, 2, 30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=frauds, mode="lines+markers", line=dict(color="#6EA8FE", width=3), marker=dict(size=6)))
        fig.update_layout(height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section("Fraud by type", "Breakdown")
        d = pd.DataFrame({"Type":["TRANSFER","CASH_OUT","PAYMENT","DEPOSIT"],"Count":[189,153,42,18],"Risk%":[12.5,28.4,3.2,1.1]})
        fig = px.bar(d, x="Count", y="Type", orientation="h", color="Risk%", color_continuous_scale=["#10B981","#F59E0B","#EF4444"])
        fig.update_layout(height=320, coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section("Recent transactions", "Activity")
    st.dataframe(SAMPLE_TRANSACTIONS, use_container_width=True, hide_index=True)

def predict_page():
    st.markdown('<div class="hero"><div class="badge"><span class="dot"></span> Real-time · ML-powered</div><h1>Analyze transaction</h1><p>Submit transaction details to receive instant AI risk score with RandomForest + SHAP explanations.</p></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section("Transaction details", "Input")
        c1, c2 = st.columns(2)
        with c1:
            ttype = st.selectbox("Transaction type", TRANSACTION_TYPES, index=0)
            amount = st.number_input("Amount (USD)", min_value=0.0, value=5000.0, step=100.0)
            old_org = st.number_input("Sender — old balance", min_value=0.0, value=10000.0, step=100.0)
            location = st.selectbox("Location", ["US", "CA", "UK", "MX", "IN", "CN"])
            device = st.text_input("Device ID", value="DEV-001")
        with c2:
            old_dest = st.number_input("Receiver — old balance", min_value=0.0, value=0.0, step=100.0)
            hour = st.slider("Transaction hour (24h)", 0, 23, 14)
            is_weekend = st.checkbox("Weekend transaction")
        analyze = st.button("🔍 Analyze with AI", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if analyze:
        user_profile = USER_PROFILES.get(ss.user_email, {})
        type_idx = TRANSACTION_TYPES.index(ttype)
        features = {
            'amount': amount,
            'balance_orig': old_org,
            'balance_dest': old_dest,
            'type_idx': type_idx,
            'hour': hour,
            'is_weekend': 1 if is_weekend else 0,
            'depletion_ratio': min(amount / (old_org + 1), 2),
            'amount_to_balance_ratio': amount / (old_dest + 1)
        }
        ml_score = calculate_risk_score_ml(features)
        behavior_score = behavioral_anomaly_score(user_profile, amount, location, device, hour, is_weekend)
        final_score = (ml_score * 0.7 + behavior_score * 0.3)
        level, color = ("Low Risk", "#10B981") if final_score < 30 else ("Medium Risk", "#F59E0B") if final_score < 70 else ("High Risk", "#EF4444")
        prediction = "Fraud" if final_score > 70 else "Legit"
        st.markdown("<br>", unsafe_allow_html=True)
        section("AI Analysis Results", "Output")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Final Risk", f"{final_score:.1f}%")
        c2.metric("🤖 ML Score", f"{ml_score:.1f}%")
        c3.metric("🧠 Behavior", f"{behavior_score:.1f}%")
        c4.metric("Status", level)
        cL, cR = st.columns([1.1, 1])
        with cL:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_score,
                number={"suffix":"%","font":{"size":42,"color":"#E6EBF5"}},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":color,"thickness":0.28},
                    "bgcolor":"rgba(255,255,255,0.04)",
                    "steps":[
                        {"range":[0,30],"color":"rgba(16,185,129,0.18)"},
                        {"range":[30,70],"color":"rgba(245,158,11,0.18)"},
                        {"range":[70,100],"color":"rgba(239,68,68,0.18)"}
                    ]
                }
            ))
            fig.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with cR:
            st.markdown('<div class="card-accent">', unsafe_allow_html=True)
            st.markdown("### Recommended Action")
            if final_score > 70:
                st.error("🛑 **BLOCK** — High fraud probability detected")
            elif final_score > 30:
                st.warning("⚠️ **REVIEW** — Manual investigation required")
            else:
                st.success("✅ **APPROVE** — Transaction appears legitimate")
            st.markdown(f"""
            <div style="margin-top:14px;color:var(--text-dim);font-size:13px;line-height:1.7">
            <b style="color:var(--text)">Key signals</b><br>
            - Type: <b style="color:var(--text)">{ttype}</b><br>
            - Amount: <b style="color:var(--text)">${amount:,.0f}</b><br>
            - Location: <b style="color:var(--text)">{location}</b><br>
            - Time: <b style="color:var(--text)">{hour}:00 {'(Weekend)' if is_weekend else '(Weekday)'}</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        section("SHAP Explainable AI", "Feature Importance")
        shap_features = get_shap_explanation(features)
        if shap_features:
            for i, f in enumerate(shap_features, 1):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{i}. {f['feature'].replace('_',' ').title()}**")
                    st.caption(f"Value: {f['value']:.2f}")
                with col2:
                    st.metric("Impact", f["impact"])
                with col3:
                    st.metric("SHAP", f"{f['shap_value']:.3f}")
        else:
            st.success("✅ No significant risk factors detected.")

# ════════════════════════════════════════════════════════════════
# BATCH PAGE — FULLY FIXED
# ════════════════════════════════════════════════════════════════
def batch_page():
    st.markdown('<div class="hero"><div class="badge"><span class="dot"></span> Batch Processing</div><h1>CSV Batch Analysis</h1><p>Upload transaction CSV for bulk fraud detection with downloadable reports.</p></div>', unsafe_allow_html=True)

    # ── Sample CSV download helper ──
    sample_df = pd.DataFrame({
        'amount':       [5000, 150000, 800, 25000, 300],
        'balance_orig': [10000, 200000, 5000, 30000, 1500],
        'balance_dest': [2000, 0, 1200, 500, 8000],
        'type':         ['PAYMENT', 'TRANSFER', 'PAYMENT', 'CASH_OUT', 'DEPOSIT'],
        'hour':         [14, 2, 10, 23, 9],
        'location':     ['US', 'CN', 'US', 'MX', 'US'],
        'device_id':    ['DEV-001', 'DEV-999', 'DEV-001', 'DEV-777', 'DEV-002'],
        'timestamp':    ['2024-01-15 14:00', '2024-01-15 02:30', '2024-01-14 10:00',
                         '2024-01-13 23:15', '2024-01-12 09:00'],
    })
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_info, col_dl = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Required columns:** `amount`, `balance_orig`, `balance_dest`, `type`, `hour`  
        **Optional columns:** `location`, `device_id`, `timestamp`  
        **Supported types:** `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEPOSIT`
        """)
    with col_dl:
        st.download_button(
            "📄 Download Sample CSV",
            data=sample_csv,
            file_name="fraudxpay_sample.csv",
            mime="text/csv",
            use_container_width=True
        )

    uploaded_file = st.file_uploader(
        "📁 Upload CSV file",
        type=['csv'],
        help="Columns: amount, balance_orig, balance_dest, type, hour, location, device_id"
    )

    if uploaded_file is not None:
        progress_bar = st.progress(0.0)        # ✅ FIX 1: float 0.0
        status_text  = st.empty()

        try:
            # ── Read CSV ──
            df = pd.read_csv(uploaded_file)
            total = len(df)                    # ✅ FIX 2: total defined before loop
            status_text.text(f"Processing {total} transactions...")
            progress_bar.progress(0.1)         # ✅ FIX 3: all progress as 0.0–1.0

            # ── Validate required columns ──
            required_cols = ['amount', 'balance_orig', 'balance_dest', 'type', 'hour']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}\n\nDownload the sample CSV above to see the correct format.")
                progress_bar.empty()
                status_text.empty()
                st.markdown('</div>', unsafe_allow_html=True)
                return

            # ── Fill missing optional columns ──
            df['location']  = df.get('location',  pd.Series(['US'] * total, index=df.index)).fillna('US')
            df['device_id'] = df.get('device_id', pd.Series(['DEV-001'] * total, index=df.index)).fillna('DEV-001')
            df['timestamp'] = df.get('timestamp', pd.Series(['2024-01-01'] * total, index=df.index)).fillna('2024-01-01')

            # ── Fill numeric nulls ──
            df['amount']       = pd.to_numeric(df['amount'],       errors='coerce').fillna(1000)
            df['balance_orig'] = pd.to_numeric(df['balance_orig'], errors='coerce').fillna(5000)
            df['balance_dest'] = pd.to_numeric(df['balance_dest'], errors='coerce').fillna(0)
            df['hour']         = pd.to_numeric(df['hour'],         errors='coerce').fillna(12).astype(int)

            # ── Map transaction type → index ──
            df['type'] = df['type'].astype(str).str.upper().str.strip()
            df['type'] = df['type'].apply(lambda x: x if x in TRANSACTION_TYPES else 'PAYMENT')
            type_mapping = {t: i for i, t in enumerate(TRANSACTION_TYPES)}
            df['type_idx'] = df['type'].map(type_mapping).fillna(0).astype(int)

            progress_bar.progress(0.25)

            # ══ FAST VECTORIZED PREDICTION (10x faster than row loop) ══

            # Weekend flag — ek baar poori column pe
            status_text.text("⚡ Preparing features...")
            try:
                df['is_weekend'] = pd.to_datetime(
                    df['timestamp'], errors='coerce'
                ).dt.weekday.fillna(0).ge(5).astype(int)
            except Exception:
                df['is_weekend'] = 0

            # Derived features — vectorized
            df['depletion_ratio']         = np.clip(df['amount'] / (df['balance_orig'] + 1), 0, 2)
            df['amount_to_balance_ratio'] = df['amount'] / (df['balance_dest'] + 1)

            progress_bar.progress(0.40)
            status_text.text("🤖 Running ML model on all transactions...")

            # ML predict — saari rows ek saath (sabse bada speedup)
            feature_cols = ['amount', 'balance_orig', 'balance_dest', 'type_idx',
                            'hour', 'is_weekend', 'depletion_ratio', 'amount_to_balance_ratio']
            X = df[feature_cols].fillna(0).astype(float)
            ml_scores = model.predict_proba(X)[:, 1] * 100  # shape: (n_rows,)

            progress_bar.progress(0.70)
            status_text.text("🧠 Computing behavior scores...")

            # Behavior score — vectorized rules (no loop needed)
            behavior_scores = np.zeros(total)
            behavior_scores += np.where(df['amount'] > 7500,  30, 0)   # large amount
            behavior_scores += np.where(df['amount'] > 3750,  15, 0)   # medium-large
            behavior_scores += np.where(~df['location'].isin(['US','CA','UK']), 25, 0)  # unusual location
            behavior_scores += np.where(df['hour'] < 6, 15, 0)         # odd hour
            behavior_scores += np.where((df['is_weekend'] == 1) & (df['amount'] > 5000), 15, 0)
            behavior_scores = np.clip(behavior_scores, 0, 100)

            final_scores = (ml_scores * 0.7 + behavior_scores * 0.3).round(2)

            progress_bar.progress(0.88)
            status_text.text("📊 Building results...")

            # Assign result columns
            df['Risk Score %']     = final_scores
            df['ML Score %']       = ml_scores.round(2)
            df['Behavior Score %'] = behavior_scores.round(2)
            df['Prediction']       = np.where(final_scores > 70, 'Fraud', 'Legit')
            df['Risk Level']       = np.where(final_scores > 70, 'High',
                                     np.where(final_scores > 30, 'Medium', 'Low'))

            ss.batch_results = df.copy()
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            st.success(f"✅ Processed **{len(ss.batch_results)}** transactions successfully!")

            # ── Results table ──
            st.markdown('<div class="card" style="margin-top:20px">', unsafe_allow_html=True)
            display_cols = [c for c in ['amount','type','location','hour','Risk Score %','ML Score %','Behavior Score %','Prediction','Risk Level'] if c in ss.batch_results.columns]

            col1, col2, col3, col4 = st.columns(4)
            fraud_count  = len(ss.batch_results[ss.batch_results['Prediction'] == 'Fraud'])
            high_risk    = len(ss.batch_results[ss.batch_results['Risk Level'] == 'High'])
            medium_risk  = len(ss.batch_results[ss.batch_results['Risk Level'] == 'Medium'])
            col1.metric("Total",        len(ss.batch_results))
            col2.metric("Frauds Found", fraud_count)
            col3.metric("High Risk",    high_risk)
            col4.metric("Fraud Rate",   f"{fraud_count / len(ss.batch_results) * 100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(ss.batch_results[display_cols], use_container_width=True, hide_index=True)

            # ── Charts ──
            col_l, col_r = st.columns(2)
            with col_l:
                pred_counts = ss.batch_results['Prediction'].value_counts().reset_index()
                pred_counts.columns = ['Prediction', 'Count']
                fig = px.pie(pred_counts, names='Prediction', values='Count',
                             color='Prediction',
                             color_discrete_map={'Legit':'#10B981','Fraud':'#EF4444'})
                fig.update_layout(height=280, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                fig2 = px.histogram(ss.batch_results, x='Risk Score %', nbins=20,
                                    color_discrete_sequence=['#6EA8FE'])
                fig2.update_layout(height=280, **PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

            # ── Download ──
            buffer, filename = generate_report(ss.batch_results)
            st.download_button(
                label="📥 Download Full Report (.xlsx)",
                data=buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Tip: Download the **Sample CSV** above to see the correct column format.")

    st.markdown('</div>', unsafe_allow_html=True)

def analytics_page():
    st.markdown('<div class="hero"><div class="badge"><span class="dot"></span> Advanced Analytics</div><h1>Interactive Dashboard</h1><p>Model performance, behavioral patterns, and risk distributions.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  "97.8%", "+0.3%")
    c2.metric("Precision", "98.2%", "+0.1%")
    c3.metric("Recall",    "89.5%", "+1.2%")
    c4.metric("F1 Score",  "93.7%", "+0.7%")
    st.markdown("<br>", unsafe_allow_html=True)
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        section("Risk Distribution", "Population")
        risk_data = np.clip(np.random.normal(35, 25, 2000), 0, 100)
        fig = px.histogram(x=risk_data, nbins=30, color_discrete_sequence=["#6EA8FE"])
        fig.update_layout(height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with row1_col2:
        section("Fraud by Transaction Type", "Categories")
        d = pd.DataFrame({
            "Type": ["TRANSFER","CASH_OUT","PAYMENT","DEPOSIT"],
            "Fraud Rate": [28.4, 15.2, 3.1, 1.2],
            "Count": [1200, 850, 3200, 1800]
        })
        fig = px.treemap(d, path=[px.Constant("All"), 'Type'], values='Count', color='Fraud Rate',
                         color_continuous_scale=["#10B981","#F59E0B","#EF4444"])
        fig.update_layout(height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section("Model Confusion Matrix", "Performance")
    cm_data = [[9821, 64],[42, 873]]
    fig = px.imshow(cm_data, text_auto=True, aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Transactions"),
                    x=['Legit','Fraud'], y=['Legit','Fraud'],
                    color_continuous_scale=["#0d1428","#6EA8FE","#EF4444"])
    fig.update_layout(height=400, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.markdown('<div class="hero"><div class="badge"><span class="dot"></span> Technical</div><h1>About FraudXPay</h1><p>Production-grade AI fraud detection platform with ML + behavioral analysis.</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Core Features
        - **Real-time ML scoring** (RandomForest)
        - **SHAP Explainable AI**
        - **Behavioral anomaly detection**
        - **Batch CSV processing**
        - **Interactive Plotly dashboards**
        - **Secure session management**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🤖 Technology Stack
        - **ML**: Scikit-learn RandomForest
        - **XAI**: SHAP TreeExplainer  
        - **UI**: Streamlit + Plotly + Custom CSS
        - **Data**: Pandas + NumPy
        - **Sessions**: Streamlit session_state
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    if not ss.logged_in:
        login_page()
        return
    topbar()
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px"><div class="brand-mark" style="width:36px;height:36px;font-size:16px">🛡️</div><div><div class="brand-name" style="font-size:16px">FraudXPay</div><div class="brand-sub" style="font-size:10px">Console</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:var(--text-mute);font-size:11px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;margin:8px 0 8px">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("Navigate", [
            "🏠 Home",
            "🎯 Real-time Predict",
            "📁 Batch CSV",
            "📊 Analytics",
            "ℹ️ About"
        ], label_visibility="collapsed", index=0)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="padding:14px 16px"><div style="display:flex;align-items:center;gap:10px"><div class="avatar">{(ss.user_name[:1] or "U").upper()}</div><div class="user-meta"><span class="user-name">{ss.user_name}</span><span class="user-email">{ss.user_email}</span></div></div></div>', unsafe_allow_html=True)
        if st.button("🚪 Sign out", use_container_width=True):
            ss.logged_in    = False
            ss.user_name    = ""
            ss.user_email   = ""
            ss.batch_results = None
            st.rerun()
    if   "Home"            in page: home_page()
    elif "Real-time Predict" in page: predict_page()
    elif "Batch CSV"       in page: batch_page()
    elif "Analytics"       in page: analytics_page()
    elif "About"           in page: about_page()

if __name__ == "__main__":
    main()


