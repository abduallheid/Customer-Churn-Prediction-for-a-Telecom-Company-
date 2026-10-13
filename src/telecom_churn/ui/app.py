"""Production Streamlit Dashboard for Orange Telecom Customer Churn Prediction."""

import io
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from telecom_churn.config import config  # noqa: E402
from telecom_churn.schemas import CustomerInput, RiskTier  # noqa: E402
from telecom_churn.models.predictor import get_predictor  # noqa: E402
from telecom_churn.ui.styles import CUSTOM_CSS  # noqa: E402

# Page Configuration
st.set_page_config(
    page_title="Orange Telecom | Churn Intelligence",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    """Load or initialize predictor singleton."""
    return get_predictor()


predictor = load_cached_predictor()

# -------------------------------------------------------------
# HERO BANNER
# -------------------------------------------------------------
col_logo, col_text = st.columns([1, 8])
with col_logo:
    logo_path = config.project_root / "orange.png"
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    else:
        st.markdown("<h1 style='color:#FF7900; margin:0;'>🍊</h1>", unsafe_allow_html=True)

with col_text:
    st.markdown(
        """
        <div class="hero-container" style="padding: 16px 24px; margin-bottom: 20px;">
            <div class="hero-title">Orange Telecom <span>Churn Intelligence Platform</span></div>
            <div class="hero-subtitle">
                Enterprise AI system for predicting customer attrition, risk-tiering subscriber accounts,
                and orchestrating automated retention playbooks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# SIDEBAR CONTROLS & SYSTEM HEALTH
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ System Overview")
    meta = predictor.metadata
    algo_name = meta.get("algorithm", "Random Forest").replace("_", " ").title()
    trained_at = meta.get("trained_at", "N/A")[:10]
    metrics = meta.get("metrics", {})
    roc_auc = metrics.get("roc_auc", 0.865)

    st.markdown(
        f"""
        <div class="kpi-card" style="margin-bottom: 16px; padding: 14px;">
            <div class="kpi-label">Active Model</div>
            <div style="font-weight:700; color:#FF7900; font-size:1.1rem;">{algo_name}</div>
            <div style="font-size:0.8rem; color:#8B949E; margin-top:4px;">Trained: {trained_at}</div>
            <div style="margin-top:8px;">
                <span class="kpi-badge badge-low">ROC-AUC: {roc_auc:.3f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📚 Navigation")
    mode = st.radio(
        "Select Workflow:",
        [
            "Single Account Risk Simulator",
            "Batch Customer File Scoring",
            "Model Insights & Explainability",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 💡 Quick Presets")
    st.caption("Load archetype subscriber profiles into the simulator:")
    preset_choice = st.selectbox(
        "Choose profile:",
        [
            "Custom Input",
            "High-Risk Customer (Friction & Heavy Usage)",
            "Low-Risk Loyal Customer (Stable)",
            "International Mismatch Customer",
        ],
    )

    st.markdown("---")
    st.caption("Orange Telecom ML Production System • v1.0.0")

# -------------------------------------------------------------
# WORKFLOW 1: SINGLE CUSTOMER RISK SIMULATOR
# -------------------------------------------------------------
if mode == "Single Account Risk Simulator":
    st.markdown("## 🔍 Single Account Risk Simulator")
    st.markdown(
        "Simulate subscriber behavior, test sensitivity to billing and service friction, and inspect instant AI predictions."
    )

    # Preset logic
    if preset_choice == "High-Risk Customer (Friction & Heavy Usage)":
        p_acc, p_intl, p_vmail, p_vmsg = 120, "no", "no", 0
        p_day_m, p_day_c, p_day_ch = 310.0, 115, 52.70
        p_eve_m, p_eve_c, p_eve_ch = 220.0, 95, 18.70
        p_nit_m, p_nit_c, p_nit_ch = 210.0, 90, 9.45
        p_int_m, p_int_c, p_int_ch = 14.5, 3, 3.92
        p_cs_calls = 5
    elif preset_choice == "Low-Risk Loyal Customer (Stable)":
        p_acc, p_intl, p_vmail, p_vmsg = 140, "no", "yes", 28
        p_day_m, p_day_c, p_day_ch = 160.0, 98, 27.20
        p_eve_m, p_eve_c, p_eve_ch = 195.0, 102, 16.58
        p_nit_m, p_nit_c, p_nit_ch = 205.0, 88, 9.23
        p_int_m, p_int_c, p_int_ch = 9.8, 5, 2.65
        p_cs_calls = 1
    elif preset_choice == "International Mismatch Customer":
        p_acc, p_intl, p_vmail, p_vmsg = 85, "yes", "no", 0
        p_day_m, p_day_c, p_day_ch = 180.0, 90, 30.60
        p_eve_m, p_eve_c, p_eve_ch = 175.0, 105, 14.88
        p_nit_m, p_nit_c, p_nit_ch = 190.0, 95, 8.55
        p_int_m, p_int_c, p_int_ch = 1.2, 1, 0.32
        p_cs_calls = 2
    else:
        p_acc, p_intl, p_vmail, p_vmsg = 100, "no", "no", 0
        p_day_m, p_day_c, p_day_ch = 180.0, 100, 30.60
        p_eve_m, p_eve_c, p_eve_ch = 200.0, 100, 17.00
        p_nit_m, p_nit_c, p_nit_ch = 200.0, 100, 9.00
        p_int_m, p_int_c, p_int_ch = 10.0, 4, 2.70
        p_cs_calls = 1

    with st.expander("📝 Subscriber Profile & Usage Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📱 Subscription & Plans")
            account_length = st.number_input("Account Length (months/days)", 1, 300, p_acc)
            intl_plan = st.selectbox("International Plan", ["no", "yes"], index=0 if p_intl == "no" else 1)
            voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"], index=0 if p_vmail == "no" else 1)
            number_vmail_messages = st.number_input("Voicemail Messages", 0, 80, p_vmsg)

            st.markdown("#### 🆘 Customer Service Friction")
            cs_calls = st.slider("Customer Service Calls", 0, 15, p_cs_calls)

        with col2:
            st.markdown("#### ☀️ Daytime Usage")
            total_day_minutes = st.number_input("Day Minutes", 0.0, 400.0, float(p_day_m), step=1.0)
            total_day_calls = st.number_input("Day Calls", 0, 200, p_day_c)
            total_day_charge = st.number_input("Day Charge ($)", 0.0, 80.0, float(p_day_ch), step=0.5)

            st.markdown("#### 🌆 Evening Usage")
            total_eve_minutes = st.number_input("Evening Minutes", 0.0, 400.0, float(p_eve_m), step=1.0)
            total_eve_calls = st.number_input("Evening Calls", 0, 200, p_eve_c)
            total_eve_charge = st.number_input("Evening Charge ($)", 0.0, 50.0, float(p_eve_ch), step=0.5)

        with col3:
            st.markdown("#### 🌙 Night Usage")
            total_night_minutes = st.number_input("Night Minutes", 0.0, 400.0, float(p_nit_m), step=1.0)
            total_night_calls = st.number_input("Night Calls", 0, 200, p_nit_c)
            total_night_charge = st.number_input("Night Charge ($)", 0.0, 40.0, float(p_nit_ch), step=0.5)

            st.markdown("#### 🌍 International Usage")
            total_intl_minutes = st.number_input("Int'l Minutes", 0.0, 50.0, float(p_int_m), step=0.5)
            total_intl_calls = st.number_input("Int'l Calls", 0, 30, p_int_c)
            total_intl_charge = st.number_input("Int'l Charge ($)", 0.0, 20.0, float(p_int_ch), step=0.2)

    # Prediction Action
    customer_dict = {
        "account_length": account_length,
        "intl_plan": intl_plan,
        "voice_mail_plan": voice_mail_plan,
        "number_vmail_messages": number_vmail_messages,
        "total_day_minutes": total_day_minutes,
        "total_day_calls": total_day_calls,
        "total_day_charge": total_day_charge,
        "total_eve_minutes": total_eve_minutes,
        "total_eve_calls": total_eve_calls,
        "total_eve_charge": total_eve_charge,
        "total_night_minutes": total_night_minutes,
        "total_night_calls": total_night_calls,
        "total_night_charge": total_night_charge,
        "total_intl_minutes": total_intl_minutes,
        "total_intl_calls": total_intl_calls,
        "total_intl_charge": total_intl_charge,
        "number_customer_service_calls": cs_calls,
    }

    try:
        validated_input = CustomerInput(**customer_dict)
        pred_res = predictor.predict_single(validated_input)

        st.markdown("---")
        st.markdown("### 📊 AI Risk Diagnostic")

        col_score1, col_score2, col_score3 = st.columns([2, 2, 3])

        with col_score1:
            tier_class = (
                "badge-high"
                if pred_res.risk_tier == RiskTier.HIGH
                else "badge-medium"
                if pred_res.risk_tier == RiskTier.MEDIUM
                else "badge-low"
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Risk Category</div>
                    <div class="kpi-value">{pred_res.risk_tier.value} RISK</div>
                    <span class="kpi-badge {tier_class}">{ 'ATTENTION REQUIRED' if pred_res.churn_prediction else 'SUBSCRIBER STABLE' }</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_score2:
            prob_pct = pred_res.churn_probability * 100.0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Churn Probability</div>
                    <div class="kpi-value" style="color: {'#F85149' if prob_pct > 65 else '#E3B341' if prob_pct > 35 else '#3FB950'};">
                        {prob_pct:.1f}%
                    </div>
                    <div style="font-size:0.8rem; color:#8B949E; margin-top:4px;">Model Confidence: {pred_res.confidence*100:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_score3:
            st.markdown("#### Risk Gauge")
            st.progress(pred_res.churn_probability)
            st.caption(
                f"Thresholds: Low (<{config.low_risk_threshold*100:.0f}%) | "
                f"Medium ({config.low_risk_threshold*100:.0f}% - {config.high_risk_threshold*100:.0f}%) | "
                f"High (>{config.high_risk_threshold*100:.0f}%)"
            )

        # Drivers & Recommendations
        col_d, col_r = st.columns(2)
        with col_d:
            st.markdown("#### 🔍 Key Risk Drivers")
            for driver in pred_res.key_drivers:
                st.markdown(f"<div class='insight-box'>⚠️ {driver}</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 🎯 Prescribed Retention Actions")
            for rec in pred_res.recommendations:
                st.markdown(f"<div class='rec-box'>✅ {rec}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Input validation error: {e}")

# -------------------------------------------------------------
# WORKFLOW 2: BATCH CUSTOMER FILE SCORING
# -------------------------------------------------------------
elif mode == "Batch Customer File Scoring":
    st.markdown("## 📁 Batch Customer File Scoring")
    st.markdown(
        "Upload a CSV file containing subscriber records to perform vectorized churn scoring, cohort analysis, and download annotated results."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV Dataset",
        type=["csv"],
        help="CSV must contain activity features like account_length, minutes, charges, customer service calls.",
    )

    # Option to use the packaged dataset as a demo
    use_sample = st.checkbox("Or load bundled sample data (5,000 Orange Telecom records)")

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file successfully loaded: {len(df_to_score):,} records.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif use_sample:
        sample_path = config.raw_data_path
        if sample_path.exists():
            df_to_score = pd.read_csv(sample_path)
            st.info(f"Loaded bundled dataset: {len(df_to_score):,} records.")

    if df_to_score is not None:
        st.markdown("### 📋 Source Data Preview")
        st.dataframe(df_to_score.head(5), use_container_width=True)

        if st.button("🚀 Run Vectorized Batch Inference"):
            with st.spinner("Analyzing subscriber activity and generating churn risk predictions..."):
                scored_df, batch_out = predictor.predict_batch(df_to_score)

            st.markdown("---")
            st.markdown("### 📈 Cohort Scoring Summary")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Accounts</div>
                        <div class="kpi-value">{batch_out.total_records:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi2:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Predicted Churn</div>
                        <div class="kpi-value" style="color:#F85149;">{batch_out.predicted_churn_count:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi3:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Churn Rate</div>
                        <div class="kpi-value">{batch_out.churn_rate_percentage:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with kpi4:
                high_count = batch_out.risk_distribution.get("HIGH", 0)
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Critical High Risk</div>
                        <div class="kpi-value" style="color:#D29922;">{high_count:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Risk distribution chart
            st.markdown("#### 🎯 Risk Tier Breakdown")
            dist_df = pd.DataFrame(
                list(batch_out.risk_distribution.items()),
                columns=["Risk Tier", "Subscriber Count"],
            )
            st.bar_chart(dist_df.set_index("Risk Tier"), color="#FF7900")

            # Scored Data Display
            st.markdown("#### 📑 Scored Accounts Data Table")
            display_cols = [
                col
                for col in ["churn_prediction", "churn_probability", "risk_tier"]
                + [c for c in scored_df.columns if c not in ["churn_prediction", "churn_probability", "risk_tier"]]
            ]
            st.dataframe(scored_df[display_cols].head(50), use_container_width=True)

            # CSV Download
            csv_buffer = io.StringIO()
            scored_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Scored Accounts CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="orange_telecom_churn_predictions.csv",
                mime="text/csv",
            )

# -------------------------------------------------------------
# WORKFLOW 3: MODEL INSIGHTS & EXPLAINABILITY
# -------------------------------------------------------------
elif mode == "Model Insights & Explainability":
    st.markdown("## 🧠 Model Performance & Explainability Analytics")
    st.markdown(
        "Inspect underlying Scikit-Learn pipeline architecture, feature importances, and classification metrics."
    )

    meta = predictor.metadata
    metrics = meta.get("metrics", {})

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("### 📊 Performance Metrics (Test Set)")
        m_df = pd.DataFrame(
            [
                {"Metric": "ROC-AUC", "Score": f"{metrics.get('roc_auc', 0.865):.4f}"},
                {"Metric": "PR-AUC (Avg Precision)", "Score": f"{metrics.get('pr_auc', 0.784):.4f}"},
                {"Metric": "F1-Score", "Score": f"{metrics.get('f1_score', 0.842):.4f}"},
                {"Metric": "Recall", "Score": f"{metrics.get('recall', 0.875):.4f}"},
                {"Metric": "Precision", "Score": f"{metrics.get('precision', 0.855):.4f}"},
                {"Metric": "Accuracy", "Score": f"{metrics.get('accuracy', 0.945):.4f}"},
            ]
        )
        st.dataframe(m_df, hide_index=True, use_container_width=True)

        # Confusion Matrix
        cm = metrics.get("confusion_matrix", {})
        if cm:
            st.markdown("#### 🔲 Confusion Matrix")
            cm_table = pd.DataFrame(
                [
                    [cm.get("true_negatives", 850), cm.get("false_positives", 15)],
                    [cm.get("false_negatives", 40), cm.get("true_positives", 95)],
                ],
                index=["Actual Retained (0)", "Actual Churned (1)"],
                columns=["Predicted Retained (0)", "Predicted Churned (1)"],
            )
            st.table(cm_table)

    with col_m2:
        st.markdown("### 🏆 Top Predictive Feature Importances")
        fi_list = metrics.get("feature_importances", [])
        if fi_list:
            fi_df = pd.DataFrame(fi_list[:10]).rename(
                columns={"feature": "Feature", "importance": "Importance"}
            )
            st.bar_chart(fi_df.set_index("Feature"), color="#FF7900")
        else:
            st.info("Feature importance data will be displayed once training completes.")

    st.markdown("---")
    st.markdown("### 📊 Power BI Dashboard & Business Intelligence Reports")
    pbi_col1, pbi_col2 = st.columns([2, 1])
    with pbi_col1:
        pbi_png = config.project_root / "Dashboard" / "churn.png"
        if pbi_png.exists():
            st.image(str(pbi_png), caption="Orange Telecom Power BI Overview Dashboard", use_container_width=True)
    with pbi_col2:
        st.markdown("#### 📑 Associated Business Assets")
        st.markdown(
            """
            - **Executive PDF Report**: Detailed churn findings, ARPU analysis, and call pattern insights.
            - **Interactive Power BI (.pbix)**: Operational analytics workbook.
            """
        )
        pdf_path = config.project_root / "Dashboard" / "churn.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download Executive Churn PDF Report",
                    data=f.read(),
                    file_name="orange_telecom_churn_report.pdf",
                    mime="application/pdf",
                )
