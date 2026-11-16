"""
Customer Segmentation Intelligence Dashboard
A professional SaaS-style analytics dashboard built with Streamlit & Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background override */
.stApp { background-color: #0e1117; }

/* ── Gradient Header ── */
.dashboard-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0d9488 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(79,70,229,.35);
}
.dashboard-header h1 {
    color: #fff; margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px;
}
.dashboard-header p {
    color: rgba(255,255,255,.82); margin: .35rem 0 0; font-size: .95rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3d 100%);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,.3);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,.25);
}
.metric-card .metric-value {
    font-size: 2rem; font-weight: 700; color: #818cf8; margin: .4rem 0 .15rem;
}
.metric-card .metric-label {
    font-size: .82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 500;
}
.metric-card .metric-icon {
    font-size: 1.5rem;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.15rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #6366f1; padding-left: .75rem;
    margin: 1.5rem 0 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131320 0%, #1a1a2e 100%);
}
[data-testid="stSidebar"] .stRadio > label { color: #c7d2fe; font-weight: 500; }

/* ── Info / Success boxes ── */
.insight-box {
    background: rgba(99,102,241,.08);
    border: 1px solid rgba(99,102,241,.3);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: .6rem 0;
    color: #c7d2fe;
    font-size: .92rem;
    line-height: 1.6;
}

/* ── Profile Cards ── */
.profile-card {
    background: linear-gradient(145deg, #1e1e2e 0%, #252540 100%);
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.profile-card h3 { color: #818cf8; margin: 0 0 .4rem; font-size: 1.1rem; }
.profile-card .stat-row { display: flex; justify-content: space-between; padding: .25rem 0; }
.profile-card .stat-label { color: #94a3b8; font-size: .85rem; }
.profile-card .stat-value { color: #e2e8f0; font-weight: 600; font-size: .85rem; }

/* Hide default Streamlit header/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Plotly chart containers */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Plotly Theme
# ──────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = ["#6366f1", "#a78bfa", "#14b8a6", "#f472b6", "#facc15", "#38bdf8", "#fb923c", "#4ade80", "#e879f9", "#f87171"]
ACCENT = "#6366f1"


# ──────────────────────────────────────────────
# Helper: load / run pipeline
# ──────────────────────────────────────────────
@st.cache_data
def load_raw_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def run_full_pipeline():
    """Execute the entire ML pipeline and cache results in session_state."""
    from src.data_preprocessing import DataPreprocessor
    from src.clustering import ClusteringEngine
    from src.cluster_evaluation import ClusterEvaluator
    from src.customer_profiling import CustomerProfiler
    from src.recommendations import RecommendationEngine

    progress = st.progress(0, text="Initializing pipeline…")

    # Step 1 – Data preprocessing
    progress.progress(10, text="Preprocessing data…")
    preprocessor = DataPreprocessor()
    pipeline_result = preprocessor.run_pipeline("data/Mall_Customers.csv")

    original_df = pipeline_result["original_df"]
    processed_df = pipeline_result["processed_df"]
    features = pipeline_result["features"]
    scaled_features = pipeline_result["scaled_features"]
    scaler = pipeline_result["scaler"]
    feature_names = pipeline_result["feature_names"]

    # Step 2 – Auto-select optimal clusters
    progress.progress(30, text="Evaluating clustering algorithms…")
    engine = ClusteringEngine()
    auto_result = engine.auto_select_optimal_clusters(scaled_features, k_range=range(2, 11))

    best_algorithm = auto_result["best_algorithm"]
    best_k = auto_result["best_k"]

    # Step 3 – Train final model
    progress.progress(55, text="Training final model…")
    model, labels = engine.train_final_model(scaled_features, best_algorithm, best_k)

    os.makedirs("models", exist_ok=True)
    engine.save_model(model, "models/clustering_model.pkl")
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Step 4 – Evaluation
    progress.progress(70, text="Evaluating clusters…")
    evaluator = ClusterEvaluator()
    eval_report = evaluator.generate_evaluation_report(scaled_features, labels, best_algorithm)
    metrics_df = evaluator.generate_metrics_dataframe(
        auto_result["kmeans_metrics"], auto_result["gmm_metrics"], k_range=range(2, 11)
    )

    # Step 5 – Profiling
    progress.progress(82, text="Building customer profiles…")
    profiler = CustomerProfiler()
    profiles = profiler.create_cluster_profiles(original_df, labels, feature_names)
    profiles = profiler.assign_segment_names(profiles)
    cluster_summary_df = profiler.get_cluster_summary_df(profiles)
    high_value_df = profiler.identify_high_value_customers(original_df, labels, top_percentile=20)

    # Step 6 – Recommendations
    progress.progress(92, text="Generating recommendations…")
    rec_engine = RecommendationEngine()
    recommendations = rec_engine.generate_segment_recommendations(profiles)
    overall_insights = rec_engine.generate_overall_insights(profiles, best_k)
    targeting = rec_engine.get_targeting_strategies(profiles)

    progress.progress(100, text="Pipeline complete ✅")

    # Persist in session_state
    st.session_state["pipeline_done"] = True
    st.session_state["original_df"] = original_df
    st.session_state["processed_df"] = processed_df
    st.session_state["features"] = features
    st.session_state["scaled_features"] = scaled_features
    st.session_state["scaler"] = scaler
    st.session_state["feature_names"] = feature_names
    st.session_state["auto_result"] = auto_result
    st.session_state["best_algorithm"] = best_algorithm
    st.session_state["best_k"] = best_k
    st.session_state["model"] = model
    st.session_state["labels"] = labels
    st.session_state["eval_report"] = eval_report
    st.session_state["metrics_df"] = metrics_df
    st.session_state["profiles"] = profiles
    st.session_state["cluster_summary_df"] = cluster_summary_df
    st.session_state["high_value_df"] = high_value_df
    st.session_state["recommendations"] = recommendations
    st.session_state["overall_insights"] = overall_insights
    st.session_state["targeting"] = targeting

    progress.empty()


def ensure_pipeline():
    """Run the pipeline if it hasn't been run yet."""
    if not st.session_state.get("pipeline_done", False):
        st.info("🚀 First run detected — executing ML pipeline. This takes a few seconds…")
        try:
            run_full_pipeline()
            st.success("Pipeline finished successfully!")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "Go to",
        [
            "📊 Executive Overview",
            "🔍 Exploratory Data Analysis",
            "⚙️ Automatic Cluster Selection",
            "👥 Customer Segmentation",
            "💡 Business Intelligence",
            "🔮 Predict New Customer",
            "📥 Download Reports",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#64748b;font-size:.75rem;'>"
        "Customer Segmentation Intelligence<br>v1.0 · Powered by ML</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """<div class="dashboard-header">
        <h1>🎯 Customer Segmentation Intelligence</h1>
        <p>AI-powered customer analytics &amp; segmentation platform</p>
    </div>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Run pipeline on first load
# ──────────────────────────────────────────────
ensure_pipeline()


# ═══════════════════════════════════════════════
#  PAGE 1 – Executive Overview
# ═══════════════════════════════════════════════
if page == "📊 Executive Overview":
    df = st.session_state["original_df"]
    labels = st.session_state["labels"]
    best_k = st.session_state["best_k"]

    # ── KPI Row ──
    avg_income = df["Annual Income (k$)"].mean()
    avg_spending = df["Spending Score (1-100)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, value in zip(
        [c1, c2, c3, c4],
        ["👤", "🔗", "💰", "🛒"],
        ["Total Customers", "Clusters Detected", "Avg Income (k$)", "Avg Spending Score"],
        [f"{len(df):,}", str(best_k), f"${avg_income:,.1f}k", f"{avg_spending:.1f}"],
    ):
        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # ── Two columns: table + pie ──
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head(10), width='stretch', hide_index=True)

    with right:
        st.markdown('<p class="section-header">Cluster Distribution</p>', unsafe_allow_html=True)
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig_pie = px.pie(
            names=[f"Cluster {i}" for i in cluster_counts.index],
            values=cluster_counts.values,
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.45,
        )
        fig_pie.update_layout(template=PLOTLY_TEMPLATE, height=370, margin=dict(t=20, b=20))
        fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(fig_pie, width='stretch')

    # ── Bar chart ──
    st.markdown('<p class="section-header">Customer Count per Cluster</p>', unsafe_allow_html=True)
    fig_bar = px.bar(
        x=[f"Cluster {i}" for i in cluster_counts.index],
        y=cluster_counts.values,
        labels={"x": "Cluster", "y": "Customer Count"},
        color=[f"Cluster {i}" for i in cluster_counts.index],
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig_bar.update_layout(template=PLOTLY_TEMPLATE, height=350, showlegend=False, margin=dict(t=20, b=40))
    st.plotly_chart(fig_bar, width='stretch')


# ═══════════════════════════════════════════════
#  PAGE 2 – Exploratory Data Analysis
# ═══════════════════════════════════════════════
elif page == "🔍 Exploratory Data Analysis":
    df = st.session_state["original_df"]

    # Age distribution
    st.markdown('<p class="section-header">Age Distribution</p>', unsafe_allow_html=True)
    fig = px.histogram(df, x="Age", nbins=25, color_discrete_sequence=[ACCENT], marginal="box")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        # Gender distribution
        st.markdown('<p class="section-header">Gender Distribution</p>', unsafe_allow_html=True)
        gender_counts = df["Gender"].value_counts()
        fig_g = px.bar(
            x=gender_counts.index,
            y=gender_counts.values,
            labels={"x": "Gender", "y": "Count"},
            color=gender_counts.index,
            color_discrete_sequence=["#a78bfa", "#14b8a6"],
        )
        fig_g.update_layout(template=PLOTLY_TEMPLATE, height=350, showlegend=False, margin=dict(t=20, b=40))
        st.plotly_chart(fig_g, width='stretch')

    with col2:
        # Annual Income distribution
        st.markdown('<p class="section-header">Annual Income Distribution</p>', unsafe_allow_html=True)
        fig_i = px.histogram(df, x="Annual Income (k$)", nbins=25, color_discrete_sequence=["#14b8a6"], marginal="box")
        fig_i.update_layout(template=PLOTLY_TEMPLATE, height=350, margin=dict(t=20, b=40))
        st.plotly_chart(fig_i, width='stretch')

    # Spending Score distribution
    st.markdown('<p class="section-header">Spending Score Distribution</p>', unsafe_allow_html=True)
    fig_s = px.histogram(df, x="Spending Score (1-100)", nbins=25, color_discrete_sequence=["#f472b6"], marginal="box")
    fig_s.update_layout(template=PLOTLY_TEMPLATE, height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig_s, width='stretch')

    col3, col4 = st.columns(2)

    with col3:
        # Correlation heatmap
        st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_h = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Purples",
            aspect="auto",
            labels=dict(color="Correlation"),
        )
        fig_h.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig_h, width='stretch')

    with col4:
        # Scatter: Income vs Spending by Gender
        st.markdown('<p class="section-header">Income vs Spending Score by Gender</p>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Gender",
            color_discrete_sequence=["#a78bfa", "#14b8a6"],
            opacity=0.75,
            size_max=10,
        )
        fig_sc.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, b=40))
        st.plotly_chart(fig_sc, width='stretch')


# ═══════════════════════════════════════════════
#  PAGE 3 – Automatic Cluster Selection
# ═══════════════════════════════════════════════
elif page == "⚙️ Automatic Cluster Selection":
    auto = st.session_state["auto_result"]
    best_algo = st.session_state["best_algorithm"]
    best_k = st.session_state["best_k"]
    metrics_df = st.session_state["metrics_df"]

    st.success(
        f"**Optimal Configuration Found:** {best_algo} with **K = {best_k}** clusters  \n"
        f"{auto.get('explanation', '')}"
    )

    kmeans_m = auto["kmeans_metrics"]
    gmm_m = auto.get("gmm_metrics", {})
    k_range = list(range(2, 11))

    # ── 2×2 grid ──
    st.markdown('<p class="section-header">Cluster Evaluation Metrics</p>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        # Elbow curve
        try:
            inertia_vals = kmeans_m["inertia"]
            fig_e = px.line(x=k_range, y=inertia_vals, markers=True, labels={"x": "K", "y": "Inertia"})
            fig_e.update_layout(template=PLOTLY_TEMPLATE, title="Elbow Curve (Inertia)", height=340, margin=dict(t=40, b=40))
            fig_e.update_traces(line_color="#6366f1")
            st.plotly_chart(fig_e, width='stretch')
        except Exception:
            st.info("Inertia data not available.")

    with r1c2:
        # Silhouette
        try:
            sil_vals = kmeans_m["silhouette_score"]
            fig_s = px.line(x=k_range, y=sil_vals, markers=True, labels={"x": "K", "y": "Silhouette Score"})
            fig_s.update_layout(template=PLOTLY_TEMPLATE, title="Silhouette Score vs K", height=340, margin=dict(t=40, b=40))
            fig_s.update_traces(line_color="#a78bfa")
            st.plotly_chart(fig_s, width='stretch')
        except Exception:
            st.info("Silhouette data not available.")

    with r2c1:
        # Davies-Bouldin
        try:
            db_vals = kmeans_m["davies_bouldin_score"]
            fig_d = px.line(x=k_range, y=db_vals, markers=True, labels={"x": "K", "y": "Davies-Bouldin Index"})
            fig_d.update_layout(template=PLOTLY_TEMPLATE, title="Davies-Bouldin Index vs K", height=340, margin=dict(t=40, b=40))
            fig_d.update_traces(line_color="#14b8a6")
            st.plotly_chart(fig_d, width='stretch')
        except Exception:
            st.info("Davies-Bouldin data not available.")

    with r2c2:
        # Calinski-Harabasz
        try:
            ch_vals = kmeans_m["calinski_harabasz_score"]
            fig_c = px.line(x=k_range, y=ch_vals, markers=True, labels={"x": "K", "y": "Calinski-Harabasz Score"})
            fig_c.update_layout(template=PLOTLY_TEMPLATE, title="Calinski-Harabasz Score vs K", height=340, margin=dict(t=40, b=40))
            fig_c.update_traces(line_color="#f472b6")
            st.plotly_chart(fig_c, width='stretch')
        except Exception:
            st.info("Calinski-Harabasz data not available.")

    # ── GMM BIC ──
    if gmm_m:
        st.markdown('<p class="section-header">GMM BIC vs Components</p>', unsafe_allow_html=True)
        try:
            bic_vals = gmm_m["bic"]
            fig_bic = px.line(x=k_range, y=bic_vals, markers=True, labels={"x": "Components", "y": "BIC"})
            fig_bic.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, b=40))
            fig_bic.update_traces(line_color="#facc15")
            st.plotly_chart(fig_bic, width='stretch')
        except Exception:
            st.info("GMM BIC data not available.")

    # ── Composite Scores ──
    composite = auto.get("composite_scores", [])
    if composite:
        st.markdown('<p class="section-header">Composite Score by Configuration</p>', unsafe_allow_html=True)
        comp_labels = [f"K={k}" for k in k_range]
        comp_values = composite
        try:
            fig_comp = px.bar(
                x=comp_labels,
                y=comp_values,
                labels={"x": "K (Number of Clusters)", "y": "Composite Score"},
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig_comp.update_layout(template=PLOTLY_TEMPLATE, height=360, margin=dict(t=20, b=40))
            st.plotly_chart(fig_comp, width='stretch')
        except Exception:
            st.info("Composite scores could not be plotted.")

    # ── Explanation ──
    st.markdown('<p class="section-header">Why This Configuration?</p>', unsafe_allow_html=True)
    explanation = auto.get("explanation", "The algorithm with the highest composite metric score was selected.")
    st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)

    # Metrics table
    if metrics_df is not None and not metrics_df.empty:
        st.markdown('<p class="section-header">Detailed Metrics Table</p>', unsafe_allow_html=True)
        st.dataframe(metrics_df, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════
#  PAGE 4 – Customer Segmentation
# ═══════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    df = st.session_state["original_df"].copy()
    labels = st.session_state["labels"]
    scaled = st.session_state["scaled_features"]
    model = st.session_state["model"]
    profiles = st.session_state["profiles"]

    df["Cluster"] = labels

    # ── Sidebar Filters ──
    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        gender_filter = st.multiselect("Gender", options=df["Gender"].unique().tolist(), default=df["Gender"].unique().tolist())
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
        income_range = st.slider(
            "Income Range (k$)",
            int(df["Annual Income (k$)"].min()),
            int(df["Annual Income (k$)"].max()),
            (int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max())),
        )
        cluster_filter = st.multiselect("Clusters", options=sorted(df["Cluster"].unique().tolist()), default=sorted(df["Cluster"].unique().tolist()))

    filtered = df[
        (df["Gender"].isin(gender_filter))
        & (df["Age"].between(*age_range))
        & (df["Annual Income (k$)"].between(*income_range))
        & (df["Cluster"].isin(cluster_filter))
    ]

    # ── Main scatter ──
    st.markdown('<p class="section-header">Income vs Spending Score by Cluster</p>', unsafe_allow_html=True)
    fig_main = px.scatter(
        filtered,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=filtered["Cluster"].astype(str),
        color_discrete_sequence=COLOR_PALETTE,
        hover_data=["Age", "Gender"],
        opacity=0.8,
        labels={"color": "Cluster"},
    )
    # Cluster centers
    try:
        if hasattr(model, "cluster_centers_"):
            centers = model.cluster_centers_
        elif hasattr(model, "means_"):
            centers = model.means_
        else:
            centers = None

        if centers is not None:
            # Centers are in scaled space – use profile averages for display
            for p in profiles:
                cid = p.get("cluster_id", p.get("cluster", None))
                avg_inc = p.get("mean_income", p.get("avg_income", None))
                avg_sp = p.get("mean_spending_score", p.get("avg_spending", None))
                if avg_inc is not None and avg_sp is not None:
                    fig_main.add_trace(
                        go.Scatter(
                            x=[avg_inc], y=[avg_sp],
                            mode="markers+text",
                            marker=dict(size=18, symbol="x", color="#fff", line=dict(width=2, color="#000")),
                            text=[f"C{cid}"],
                            textposition="top center",
                            textfont=dict(color="#fff", size=11),
                            showlegend=False,
                        )
                    )
    except Exception:
        pass

    fig_main.update_layout(template=PLOTLY_TEMPLATE, height=500, margin=dict(t=20, b=40))
    st.plotly_chart(fig_main, width='stretch')

    # ── 3D Scatter ──
    st.markdown('<p class="section-header">3D View: Age × Income × Spending Score</p>', unsafe_allow_html=True)
    fig_3d = px.scatter_3d(
        filtered,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color=filtered["Cluster"].astype(str),
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.8,
        labels={"color": "Cluster"},
    )
    fig_3d.update_layout(template=PLOTLY_TEMPLATE, height=550, margin=dict(t=20, b=20))
    st.plotly_chart(fig_3d, width='stretch')

    # ── Filtered Table ──
    st.markdown('<p class="section-header">Filtered Customer Data</p>', unsafe_allow_html=True)
    st.dataframe(filtered, width='stretch', hide_index=True)

    # ── Cluster Statistics ──
    st.markdown('<p class="section-header">Cluster Statistics</p>', unsafe_allow_html=True)
    cluster_stats = (
        df.groupby("Cluster")
        .agg(
            Count=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income=("Annual Income (k$)", "mean"),
            Avg_Spending=("Spending Score (1-100)", "mean"),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(cluster_stats, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════
#  PAGE 5 – Business Intelligence
# ═══════════════════════════════════════════════
elif page == "💡 Business Intelligence":
    profiles = st.session_state["profiles"]
    recommendations = st.session_state["recommendations"]
    overall_insights = st.session_state["overall_insights"]
    high_value_df = st.session_state["high_value_df"]
    targeting = st.session_state["targeting"]
    df = st.session_state["original_df"].copy()
    labels = st.session_state["labels"]
    df["Cluster"] = labels

    # ── Segment Profile Cards ──
    st.markdown('<p class="section-header">Customer Segment Profiles</p>', unsafe_allow_html=True)

    for idx, profile in enumerate(profiles):
        seg_name = profile.get("segment_name", f"Segment {idx}")
        cluster_id = profile.get("cluster_id", profile.get("cluster", idx))
        count = profile.get("count", profile.get("customer_count", 0))
        pct = profile.get("percentage", round(count / len(df) * 100, 1) if len(df) > 0 else 0)
        avg_age = profile.get("mean_age", profile.get("avg_age", 0))
        avg_inc = profile.get("mean_income", profile.get("avg_income", 0))
        avg_sp = profile.get("mean_spending_score", profile.get("avg_spending", 0))
        gender_split = profile.get("gender_split", profile.get("gender_distribution", {}))
        # Build gender split from male_count/female_count if not directly available
        if not gender_split and "male_count" in profile:
            gender_split = {"Male": profile["male_count"], "Female": profile["female_count"]}
        interpretation = ""

        rec = recommendations[idx] if idx < len(recommendations) else {}
        strategies = rec.get("strategy", rec.get("marketing_strategies", []))
        channels = rec.get("channels", rec.get("target_channels", []))
        interpretation = rec.get("interpretation", profile.get("interpretation", "—"))

        with st.expander(f"🏷️ {seg_name}  (Cluster {cluster_id}  ·  {count} customers  ·  {pct}%)", expanded=idx == 0):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Avg Age", f"{avg_age:.1f}")
            mc2.metric("Avg Income", f"${avg_inc:.1f}k" if isinstance(avg_inc, (int, float)) else str(avg_inc))
            mc3.metric("Avg Spending", f"{avg_sp:.1f}" if isinstance(avg_sp, (int, float)) else str(avg_sp))
            mc4.metric("Customers", f"{count}")

            if gender_split:
                gs_text = " · ".join(f"{k}: {v}" for k, v in gender_split.items()) if isinstance(gender_split, dict) else str(gender_split)
                st.caption(f"**Gender Split:** {gs_text}")

            if interpretation:
                st.markdown(f'<div class="insight-box">📌 <strong>Interpretation:</strong> {interpretation}</div>', unsafe_allow_html=True)

            if strategies:
                st.markdown("**🎯 Marketing Strategies:**")
                for s in strategies:
                    st.markdown(f"- {s}")

            if channels:
                st.markdown("**📡 Target Channels:** " + ", ".join(channels) if isinstance(channels, list) else str(channels))

    # ── Overall Insights ──
    st.markdown('<p class="section-header">Overall Business Insights</p>', unsafe_allow_html=True)
    if overall_insights:
        for insight in overall_insights:
            st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)

    # ── High-Value Customers ──
    st.markdown('<p class="section-header">High-Value Customers (Top 20%)</p>', unsafe_allow_html=True)
    if high_value_df is not None and not high_value_df.empty:
        st.dataframe(high_value_df, width='stretch', hide_index=True)
    else:
        st.info("No high-value customer data available.")

    # ── Radar Chart ──
    st.markdown('<p class="section-header">Cluster Comparison Radar</p>', unsafe_allow_html=True)
    try:
        categories = ["Age", "Income", "Spending"]
        fig_radar = go.Figure()

        for idx, profile in enumerate(profiles):
            seg_name = profile.get("segment_name", f"Cluster {idx}")
            vals = [
                profile.get("mean_age", profile.get("avg_age", 0)),
                profile.get("mean_income", profile.get("avg_income", 0)),
                profile.get("mean_spending_score", profile.get("avg_spending", 0)),
            ]
            # Normalise to 0-1 for radar readability
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=seg_name,
                    line_color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                    opacity=0.65,
                )
            )

        fig_radar.update_layout(
            template=PLOTLY_TEMPLATE,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, color="#64748b"),
                angularaxis=dict(color="#94a3b8"),
            ),
            height=480,
            margin=dict(t=40, b=40),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, width='stretch')
    except Exception as e:
        st.warning(f"Could not render radar chart: {e}")


# ═══════════════════════════════════════════════
#  PAGE 6 – Predict New Customer
# ═══════════════════════════════════════════════
elif page == "🔮 Predict New Customer":
    profiles = st.session_state["profiles"]

    st.markdown('<p class="section-header">Predict Cluster for a New Customer</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            new_age = st.slider("Age", 18, 70, 30)
            new_gender = st.selectbox("Gender", ["Male", "Female"])
        with fc2:
            new_income = st.slider("Annual Income (k$)", 15, 137, 60)
            new_spending = st.slider("Spending Score (1-100)", 1, 100, 50)
        submitted = st.form_submit_button("🔮 Predict Segment", use_container_width=True)

    if submitted:
        try:
            # Load scaler & model
            scaler = st.session_state.get("scaler")
            model = st.session_state.get("model")

            if scaler is None:
                with open("models/scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
            if model is None:
                with open("models/clustering_model.pkl", "rb") as f:
                    model = pickle.load(f)

            # Build feature vector (same order as training)
            gender_encoded = 1 if new_gender == "Male" else 0
            feature_names = st.session_state.get("feature_names", [])

            # Attempt to construct features in the right order
            # Common orderings used by the preprocessor:
            #   [Age, Annual Income (k$), Spending Score (1-100), Gender_Encoded]
            #   or [Gender_Encoded, Age, Annual Income (k$), Spending Score (1-100)]
            # We build a dict and align to feature_names if available
            feature_map = {
                "Age": new_age,
                "Annual Income (k$)": new_income,
                "Spending Score (1-100)": new_spending,
                "Gender_Encoded": gender_encoded,
                "Gender_Male": gender_encoded,
            }

            if feature_names:
                input_vals = [feature_map.get(fn, 0) for fn in feature_names]
            else:
                input_vals = [new_age, new_income, new_spending, gender_encoded]

            input_arr = np.array(input_vals).reshape(1, -1)
            input_scaled = scaler.transform(input_arr)

            if hasattr(model, "predict"):
                predicted_label = model.predict(input_scaled)[0]
            else:
                predicted_label = model.predict(input_scaled)[0]

            # Find matching profile
            profile_match = None
            for p in profiles:
                cid = p.get("cluster_id", p.get("cluster", None))
                if cid == predicted_label:
                    profile_match = p
                    break

            seg_name = profile_match.get("segment_name", f"Cluster {predicted_label}") if profile_match else f"Cluster {predicted_label}"
            interpretation = ""

            # Get recommendation
            recommendations = st.session_state.get("recommendations", [])
            rec_match = None
            if recommendations:
                for r in recommendations:
                    rcid = r.get("cluster_id", r.get("cluster", None))
                    if rcid == predicted_label:
                        rec_match = r
                        break
                if rec_match is None and predicted_label < len(recommendations):
                    rec_match = recommendations[predicted_label]

            if rec_match:
                interpretation = rec_match.get("interpretation", "")

            st.markdown("---")
            rc1, rc2 = st.columns([1, 2])

            with rc1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-icon">🏷️</div>
                        <div class="metric-value">Cluster {predicted_label}</div>
                        <div class="metric-label">{seg_name}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with rc2:
                if interpretation:
                    st.markdown(f'<div class="insight-box">📌 <strong>Business Interpretation:</strong><br>{interpretation}</div>', unsafe_allow_html=True)
                if rec_match:
                    strategies = rec_match.get("strategy", rec_match.get("marketing_strategies", []))
                    if strategies:
                        st.markdown("**🎯 Recommended Marketing Strategies:**")
                        for s in strategies:
                            st.markdown(f"- {s}")

            # ── Show on scatter ──
            st.markdown('<p class="section-header">New Customer on Segmentation Map</p>', unsafe_allow_html=True)
            df_plot = st.session_state["original_df"].copy()
            df_plot["Cluster"] = st.session_state["labels"]

            fig_pred = px.scatter(
                df_plot,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                color=df_plot["Cluster"].astype(str),
                color_discrete_sequence=COLOR_PALETTE,
                opacity=0.5,
                labels={"color": "Cluster"},
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=[new_income],
                    y=[new_spending],
                    mode="markers+text",
                    marker=dict(size=20, color="#facc15", symbol="star", line=dict(width=2, color="#fff")),
                    text=["NEW"],
                    textposition="top center",
                    textfont=dict(color="#facc15", size=13, family="Inter"),
                    name="New Customer",
                    showlegend=True,
                )
            )
            fig_pred.update_layout(template=PLOTLY_TEMPLATE, height=500, margin=dict(t=20, b=40))
            st.plotly_chart(fig_pred, width='stretch')

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════
#  PAGE 7 – Download Reports
# ═══════════════════════════════════════════════
elif page == "📥 Download Reports":
    df = st.session_state["original_df"].copy()
    labels = st.session_state["labels"]
    df["Cluster"] = labels
    cluster_summary_df = st.session_state["cluster_summary_df"]
    metrics_df = st.session_state["metrics_df"]

    # ── 1. Full segmented dataset ──
    st.markdown('<p class="section-header">Customer Segmentation Results</p>', unsafe_allow_html=True)
    st.dataframe(df.head(20), width='stretch', hide_index=True)
    st.download_button(
        "⬇️  Download Full Segmented Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segmentation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    # ── 2. Cluster summary ──
    st.markdown('<p class="section-header">Cluster Summary</p>', unsafe_allow_html=True)
    if cluster_summary_df is not None and not cluster_summary_df.empty:
        st.dataframe(cluster_summary_df, width='stretch', hide_index=True)
        st.download_button(
            "⬇️  Download Cluster Summary (CSV)",
            data=cluster_summary_df.to_csv(index=False).encode("utf-8"),
            file_name="cluster_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Cluster summary not available.")

    st.markdown("---")

    # ── 3. Evaluation metrics ──
    st.markdown('<p class="section-header">Evaluation Metrics</p>', unsafe_allow_html=True)
    if metrics_df is not None and not metrics_df.empty:
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        st.download_button(
            "⬇️  Download Evaluation Metrics (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="evaluation_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Evaluation metrics not available.")

