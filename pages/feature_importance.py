# pages/feature_importance.py
import streamlit as st
import pandas as pd
import joblib, json
import shap
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Feature Importance", layout="wide", page_icon="📊")

# ---- STYLE (small) ----
st.markdown(
    """
    <style>
      .stButton>button { background-color: #00C9A7; color: white; }
      .block-container { padding-top: 2rem; padding-left:3rem; padding-right:3rem; }
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- PATHS ----
MODEL_PATH = "artifacts/model.joblib"
META_PATH = "artifacts/metadata.json"
DATA_PATH = "data/greendestination.csv"
LOGO_PATH = "data/logo.png"

# ---- CACHING: load resources once ----
@st.cache_resource(show_spinner=False)
def load_model_and_meta(model_path=MODEL_PATH, meta_path=META_PATH):
    m = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return m, meta

@st.cache_data(show_spinner=False)
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

# ---- UI HEADER ----
col1, col2 = st.columns([1,4])
with col1:
    st.image(LOGO_PATH, width=120)
with col2:
    st.markdown("<h1 style='color:#00C9A7; margin-bottom:0;'>📊 Feature Importance Dashboard</h1>", unsafe_allow_html=True)
    st.write("Understand which features contribute most to the model's attrition predictions.")

st.markdown("---")

# ---- LOAD MODEL + DATA (cached) ----
model, meta = load_model_and_meta()
df = load_data()

cat_cols = meta.get("cat_cols", [])
num_cols = meta.get("num_cols", [])
all_features = cat_cols + num_cols

# ---- INTERACTIVE OPTIONS ----
st.sidebar.header("Explainability Controls")
sample_size = st.sidebar.slider("SHAP sample size (rows)", min_value=100, max_value=1000, value=300, step=50)
top_k = st.sidebar.slider("Show top features", min_value=5, max_value=30, value=12, step=1)
explain_method = st.sidebar.selectbox("Explainer (fast → slow)", ["Linear (recommended)", "Tree/Kernel (advanced)"])

# ---- Prepare sample (deterministic) ----
X = df[all_features].copy()
sample = X.sample(n=min(sample_size, len(X)), random_state=42)

# ---- Preprocess once (cached) ----
@st.cache_data(show_spinner=False)
def transform_sample(_model, sample_df):
    preprocess = _model.named_steps["preprocess"]
    X_transformed = preprocess.transform(sample_df)
    feature_names = preprocess.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    return X_transformed_df


with st.spinner("Preparing data and computing explanations — this may take a few seconds..."):
    X_transformed_df = transform_sample(model, sample)

# ---- Compute SHAP values (cached) ----
@st.cache_data(show_spinner=False)
def compute_shap(_clf, X_transformed_df, method="Linear"):
    # prefer LinearExplainer for linear models (fast)
    if method == "Linear":
        explainer = shap.LinearExplainer(_clf, X_transformed_df, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_transformed_df)
    else:
        # fallback to Explainer which may be slower but supports many models
        explainer = shap.Explainer(_clf, X_transformed_df)
        shap_vals = explainer(X_transformed_df).values
    return shap_vals

clf = model.named_steps["clf"]
shap_vals = compute_shap(clf, X_transformed_df, method="Linear" if explain_method.startswith("Linear") else "Other")

# shap_vals may be list-like (two classes) or ndarray; convert to consistent 2D abs-mean
if isinstance(shap_vals, list):  # multiclass -> take mean absolute across classes
    arr = np.mean(np.abs(np.stack(shap_vals, axis=0)), axis=0)
else:
    arr = np.abs(shap_vals)

mean_abs = np.mean(arr, axis=0)
importance_df = pd.DataFrame({
    "feature": X_transformed_df.columns,
    "importance": mean_abs
}).sort_values("importance", ascending=False)

# ---- MAIN PLOT: top K horizontal bar (Plotly) ----
st.markdown("### 💡 Top Influential Features")
top_df = importance_df.head(top_k).iloc[::-1]  # reverse for horizontal bar (small to large)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=top_df["importance"],
    y=top_df["feature"],
    orientation="h",
    marker=dict(
        color=top_df["importance"],
        colorscale="Tealrose",
        showscale=True,
        colorbar=dict(title="Importance"),
        line=dict(color="#0E1117", width=1)
    )
))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    xaxis_title="Mean |SHAP value| (impact)",
    yaxis_title="Feature",
    height=60 + top_k * 30,
    margin=dict(l=250, r=50, t=30, b=30),
    font=dict(color="#FAFAFA")
)

st.plotly_chart(fig, use_container_width=True)

# ---- DATA EXPORT ----
csv = importance_df.to_csv(index=False)
st.download_button("📥 Download full importance CSV", data=csv, file_name="feature_importance.csv", mime="text/csv")

# ---- OPTIONAL: compact SHAP summary (small sample) ----
with st.expander("🔍 Show SHAP summary plot (be patient)"):
    st.write("SHAP summary (be cautious: drawing large dotplots can be slow).")
    try:
        import matplotlib.pyplot as plt
        fig2, ax2 = plt.subplots(figsize=(10, min(12, max(6, top_k/1.5))))
        shap.summary_plot(shap_vals if not isinstance(shap_vals, list) else shap_vals[0],
                          X_transformed_df,
                          show=False, max_display=top_k)
        st.pyplot(fig2)
    except Exception as e:
        st.error("Could not render Matplotlib SHAP plot here. Use the bar chart above.")

# ---- OPTIONAL: full table ----
with st.expander("📋 Full importance table"):
    st.dataframe(importance_df, use_container_width=True)

st.markdown("""<hr style='border:1px solid #00C9A7;'>
<p style='text-align:center; color:#A9A9A9; font-size:13px;'>Built with ❤️ using Streamlit | © 2025 SharonGadi</p>""", unsafe_allow_html=True)
