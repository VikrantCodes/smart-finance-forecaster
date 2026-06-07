"""
streamlit_app/app.py
Smart Personal Finance Forecaster — Streamlit Frontend

Run:
    streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils.predict import predict_single

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Finance Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stMetric label { font-size: 14px; color: #aaa; }
    .stMetric [data-testid="metric-container"] { background: #1e1e2e; border-radius: 10px; padding: 12px; }
    .tip-box {
        background: #0d3b1e;
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("Smart Personal Finance Forecaster")
st.divider()

# ── Session state for prediction history ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar — Inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋Your Financial Details")

    st.subheader("Personal Info")
    income      = st.number_input("Monthly Income (₹)", 5000, 500000, 44637, step=1000)
    age         = st.slider("Age", 18, 65, 35)
    dependents  = st.slider("Number of Dependents", 0, 6, 1)
    occupation  = st.selectbox("Occupation", ["Professional", "Retired", "Self_Employed", "Student"])
    city_tier   = st.selectbox("City Tier", ["Tier_1", "Tier_2", "Tier_3"])

    st.divider()
    st.subheader("Monthly Expenses (₹)")

    rent            = st.number_input("Rent",           0, 100000, 13391, step=500)
    loan_repayment  = st.number_input("Loan Repayment", 0, 100000, 0,     step=500)
    insurance       = st.number_input("Insurance",      0, 50000,  2206,  step=100)
    groceries       = st.number_input("Groceries",      0, 50000,  6658,  step=200)
    transport       = st.number_input("Transport",      0, 30000,  2636,  step=200)
    eating_out      = st.number_input("Eating Out",     0, 30000,  1651,  step=200)
    entertainment   = st.number_input("Entertainment",  0, 20000,  1536,  step=200)
    utilities       = st.number_input("Utilities",      0, 20000,  2911,  step=200)
    healthcare      = st.number_input("Healthcare",     0, 20000,  1546,  step=200)
    education       = st.number_input("Education",      0, 50000,  0,     step=500)
    miscellaneous   = st.number_input("Miscellaneous",  0, 20000,  831,   step=100)

    st.divider()
    predict_btn = st.button("Predict", use_container_width=True, type="primary")

# ── Main content ─────────────────────────────────────────────────────────────────
col_main, col_history = st.columns([3, 2])

with col_main:
    if predict_btn:
        input_data = {
            "Income": income, "Age": age, "Dependents": dependents,
            "Occupation": occupation, "City_Tier": city_tier,
            "Rent": rent, "Loan_Repayment": loan_repayment, "Insurance": insurance,
            "Groceries": groceries, "Transport": transport, "Eating_Out": eating_out,
            "Entertainment": entertainment, "Utilities": utilities,
            "Healthcare": healthcare, "Education": education, "Miscellaneous": miscellaneous,
        }

        with st.spinner("Running model …"):
            try:
                result = predict_single(input_data)

                disposable = result["predicted_disposable_income"]
                total_exp  = result["total_expense"]
                exp_ratio  = result["expense_ratio_pct"]
                sug_save   = result["suggested_savings"]

                # Store in history
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Income": income,
                    "Total Expense": total_exp,
                    "Disposable": disposable,
                })

                # ── KPI row ─────────────────────────────────────────────────────
                st.subheader("📊 Prediction Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("💵 Disposable Income", f"₹{disposable:,.0f}")
                c2.metric("📉 Total Expenses",    f"₹{total_exp:,.0f}")
                c3.metric("📈 Expense Ratio",     f"{exp_ratio}%")

                # ── Savings tip ─────────────────────────────────────────────────
                health_pct = disposable / income * 100 if income > 0 else 0
                if health_pct >= 25:
                    tip_color, tip_icon, tip_msg = "#2ecc71", "🟢", "Great financial health! Consider investing your surplus."
                elif health_pct >= 10:
                    tip_color, tip_icon, tip_msg = "#f39c12", "🟡", "Moderate. Try reducing variable expenses by 10%."
                else:
                    tip_color, tip_icon, tip_msg = "#e74c3c", "🔴", "High expense pressure. Review fixed commitments like Rent/Loans."

                st.markdown(f"""
                <div class="tip-box">
                    {tip_icon} <strong>Financial Health:</strong> {health_pct:.1f}% of income is disposable.<br>
                    💡 {tip_msg}<br>
                    🎯 Suggested savings target: <strong>₹{sug_save:,.0f}/month</strong>
                </div>
                """, unsafe_allow_html=True)

                # ── Expense donut chart ─────────────────────────────────────────
                st.subheader("🧩 Expense Breakdown")
                expense_labels = [
                    "Rent", "Loan", "Insurance", "Groceries", "Transport",
                    "Eating Out", "Entertainment", "Utilities", "Healthcare",
                    "Education", "Misc", "Disposable",
                ]
                expense_values = [
                    rent, loan_repayment, insurance, groceries, transport,
                    eating_out, entertainment, utilities, healthcare,
                    education, miscellaneous, max(0, disposable),
                ]
                fig_donut = px.pie(
                    values=expense_values, names=expense_labels,
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig_donut.update_traces(textposition="inside", textinfo="percent+label")
                fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0))
                st.plotly_chart(fig_donut, use_container_width=True)

                # ── Gauge chart ─────────────────────────────────────────────────
                st.subheader("🎯 Disposable Income Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=disposable,
                    number={"prefix": "₹", "valueformat": ",.0f"},
                    delta={"reference": income * 0.20, "valueformat": ",.0f"},
                    title={"text": "Predicted Disposable Income"},
                    gauge={
                        "axis": {"range": [0, income]},
                        "bar": {"color": tip_color},
                        "steps": [
                            {"range": [0,            income * 0.10], "color": "#e74c3c"},
                            {"range": [income * 0.10, income * 0.25], "color": "#f39c12"},
                            {"range": [income * 0.25, income],        "color": "#2ecc71"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": income * 0.20},
                    },
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=30, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

            except FileNotFoundError as e:
                st.error(f"⚠️ {e}")
                st.info("Run `python train.py` from the project root first, then reload this page.")

    else:
        st.info("👈 Fill in your financial details in the sidebar, then click **Predict Disposable Income**.")
        st.markdown("""
        ### How it works
        1. Enter your monthly income and expense breakdown
        2. The ML model predicts your **disposable income**
        3. Get a financial health score and savings tips

        ### Models used
        - **LightGBM** — gradient boosting, fast and accurate
        - **CatBoost** — robust on financial tabular data
        - Best model is auto-selected by lowest RMSE
        """)


# ── Prediction History ────────────────────────────────────────────────────────────
with col_history:
    st.subheader("🕒 Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # History line chart
        if len(hist_df) > 1:
            fig_hist = px.line(
                hist_df, x="Time", y="Disposable",
                markers=True, title="Disposable Income Trend",
                labels={"Disposable": "₹ Disposable Income"},
            )
            fig_hist.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        # Average error monitoring (simple MLOps monitoring)
        avg_disp = hist_df["Disposable"].mean()
        avg_exp  = hist_df["Total Expense"].mean()
        st.divider()
        st.markdown("**📡 Session Monitoring**")
        st.metric("Avg Predicted Disposable", f"₹{avg_disp:,.0f}")
        st.metric("Avg Total Expense",         f"₹{avg_exp:,.0f}")
    else:
        st.caption("No predictions yet. Run a prediction to see history here.")

# ── Footer ────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Smart Finance Forecaster · Built with LightGBM + CatBoost + MLflow + Streamlit · Deployed on Hugging Face Spaces")
