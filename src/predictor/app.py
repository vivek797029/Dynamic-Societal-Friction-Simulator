"""
DSFS Streamlit Web Interface
Professional interactive dashboard for the Dynamic Society Friction Simulator.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np

from src.predictor.core import DSFSPredictor
from src.predictor.data.historical_cases import INDICATOR_METADATA, HISTORICAL_CASES


# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DSFS — Dynamic Society Friction Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #1a1a2e;
        text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; text-align: center; margin-bottom: 2rem;
    }
    .risk-critical { background: #ff4444; color: white; padding: 20px; border-radius: 12px; text-align: center; }
    .risk-high { background: #ff8800; color: white; padding: 20px; border-radius: 12px; text-align: center; }
    .risk-medium { background: #ffcc00; color: #333; padding: 20px; border-radius: 12px; text-align: center; }
    .risk-low { background: #00cc66; color: white; padding: 20px; border-radius: 12px; text-align: center; }
    .metric-card {
        background: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 4px solid #1a73e8; margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px; padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Predictor ────────────────────────────────────
@st.cache_resource
def load_predictor():
    predictor = DSFSPredictor(model_dir="models")
    predictor.train(n_samples=10000)
    return predictor


def get_risk_color(level):
    colors = {"CRITICAL": "#ff4444", "HIGH": "#ff8800", "MEDIUM": "#ffcc00", "LOW": "#00cc66"}
    return colors.get(level, "#666")


def get_risk_emoji(level):
    emojis = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    return emojis.get(level, "⚪")


# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Conditions")
    st.markdown("Adjust socioeconomic indicators:")

    st.markdown("### 📊 Economic")
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 40.0, 8.0, 0.5)
    inflation = st.slider("Inflation Rate (%)", 0.0, 60.0, 5.0, 0.5)
    gdp_growth = st.slider("GDP Growth (%)", -15.0, 15.0, 2.0, 0.5)
    food_index = st.slider("Food Price Index", 80.0, 200.0, 110.0, 1.0)
    poverty = st.slider("Poverty Rate (%)", 0.0, 60.0, 20.0, 1.0)

    st.markdown("### 🏛️ Political")
    stability = st.slider("Political Stability (0-1)", 0.0, 1.0, 0.5, 0.05)
    corruption = st.slider("Corruption Index (0-100)", 0, 100, 40, 1)
    press_freedom = st.slider("Press Freedom (0-1)", 0.0, 1.0, 0.5, 0.05)

    st.markdown("### 👥 Social")
    gini = st.slider("Gini Coefficient (0-1)", 0.0, 0.70, 0.35, 0.01)
    youth_bulge = st.slider("Youth Bulge (%)", 20.0, 80.0, 45.0, 1.0)
    urbanization = st.slider("Urbanization (%)", 10.0, 95.0, 50.0, 1.0)
    internet = st.slider("Internet Penetration (%)", 5.0, 99.0, 50.0, 1.0)
    ethnic_frac = st.slider("Ethnic Fractionalization (0-1)", 0.0, 1.0, 0.30, 0.05)
    military = st.slider("Military Expenditure (% GDP)", 0.0, 10.0, 2.0, 0.1)

    description = st.text_area("Describe the situation (optional):", "")

    predict_btn = st.button("🔮 PREDICT RISK", type="primary", use_container_width=True)


# ─── Build indicators dict ────────────────────────────────────
indicators = {
    "unemployment_rate": unemployment,
    "inflation_rate": inflation,
    "gdp_growth": gdp_growth,
    "food_price_index": food_index,
    "poverty_rate": poverty,
    "political_stability": stability,
    "corruption_index": corruption,
    "press_freedom_index": press_freedom,
    "gini_coefficient": gini,
    "youth_bulge_pct": youth_bulge,
    "urbanization_rate": urbanization,
    "internet_penetration": internet,
    "ethnic_fractionalization": ethnic_frac,
    "military_expenditure_pct": military
}


# ─── MAIN CONTENT ─────────────────────────────────────────────
st.markdown('<p class="main-header">🔮 Dynamic Society Friction Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hybrid ML System — LGBM + CNN-LSTM + SBERT + GCRI | NOT Generative AI</p>',
            unsafe_allow_html=True)

if predict_btn:
    predictor = load_predictor()

    with st.spinner("Running 6-layer prediction pipeline..."):
        result = predictor.predict(indicators, description)

    # ─── RISK SCORE HEADER ────────────────────────────────────
    risk_score = result["prediction"]["risk_score"]
    risk_level = result["prediction"]["risk_level"]
    confidence = result["confidence"]["confidence_score"]

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_class = f"risk-{risk_level.lower()}"
        st.markdown(f"""
        <div class="{risk_class}">
            <h1 style="margin:0; font-size:3rem;">{risk_score}%</h1>
            <h3 style="margin:0;">{get_risk_emoji(risk_level)} {risk_level} RISK</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Confidence", f"{confidence}%",
                   delta=result["confidence"]["confidence_level"])
        st.metric("Model Used", result["prediction"]["model_used"])

    with col3:
        esc = result["escalation"]
        st.metric("Current Stage", esc["current_stage"].replace("_", " ").title())
        st.metric("Trend", esc["trend"].replace("_", " ").title())

    st.divider()

    # ─── TABS ─────────────────────────────────────────────────
    tabs = st.tabs(["📈 Escalation", "📜 Historical", "🧠 Confidence",
                     "🔄 What-If", "📋 Policy", "📊 Full Report"])

    # TAB 1: Escalation
    with tabs[0]:
        st.subheader("Multi-Horizon Risk Trajectory")
        traj = result["escalation"]["risk_trajectory"]

        col1, col2, col3 = st.columns(3)
        col1.metric("1 Month", f"{traj['1_month']}%",
                     delta=f"{traj['1_month'] - risk_score:+.1f}%")
        col2.metric("3 Months", f"{traj['3_months']}%",
                     delta=f"{traj['3_months'] - risk_score:+.1f}%")
        col3.metric("6 Months", f"{traj['6_months']}%",
                     delta=f"{traj['6_months'] - risk_score:+.1f}%")

        st.subheader("Cascade Timeline")
        for step in result["cascade_timeline"]:
            emoji = "🟢" if step["projected_risk"] < 40 else "🟡" if step["projected_risk"] < 60 else "🟠" if step["projected_risk"] < 80 else "🔴"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{emoji} {step['stage']}</strong> — {step['timeframe']} from now<br>
                <small>{step['description']}</small><br>
                <strong>Projected Risk: {step['projected_risk']}%</strong> | Probability: {step['probability']}%
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: Historical Matches
    with tabs[1]:
        st.subheader("Historical Parallels")
        for match in result["historical_parallels"]:
            with st.expander(f"🔗 {match['name']} — {match['similarity_pct']}% Similar"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Severity", match["severity"].upper())
                col2.metric("Duration", f"{match['duration_days']} days")
                col3.metric("Fatalities", f"{match['fatalities']:,}")

                st.write(f"**Type:** {match['type'].replace('_', ' ').title()}")
                st.write(f"**Resolution:** {match['resolution'].replace('_', ' ').title()}")
                st.write(f"**Outcome:** {match['outcome']}")
                st.write(f"**Triggers:** {', '.join(match['triggers'])}")
                st.write(f"**Escalation:** {' → '.join(match['escalation_path'])}")

    # TAB 3: Confidence
    with tabs[2]:
        st.subheader("Confidence Analysis")
        conf = result["confidence"]

        st.progress(int(conf["confidence_score"]))
        st.markdown(f"**Overall Confidence: {conf['confidence_score']}% ({conf['confidence_level']})**")

        st.markdown("#### Factor Breakdown")
        for factor, score in conf["factor_scores"].items():
            weight = conf["factor_weights"].get(factor, 0)
            st.markdown(f"**{factor.replace('_', ' ').title()}**: {score}% (weight: {weight})")
            st.progress(int(min(score, 100)))

        if conf["reasoning"]:
            st.markdown("#### Reasoning")
            for reason in conf["reasoning"]:
                strength_emoji = "🟢" if reason["strength"] == "strong" else "🟡"
                st.markdown(f"{strength_emoji} **{reason['factor']}** ({reason['strength']}): {reason['detail']}")

    # TAB 4: What-If
    with tabs[3]:
        st.subheader("What-If Intervention Analysis")

        if result["interventions"]:
            st.markdown("#### Top Recommended Interventions")
            for rec in result["interventions"]["recommended_interventions"][:5]:
                st.markdown(f"**{rec['name']}** — Risk reduction: **{rec['risk_reduction']:.1f}pts** | "
                           f"Cost: {rec['cost']} | Speed: {rec['speed']} | "
                           f"Effectiveness: {rec['effectiveness']*100:.0f}%")

            st.divider()
            st.markdown("#### Detailed Analysis of Top 3")
            for analysis in result["interventions"]["top_3_analysis"]:
                interv = analysis["intervention"]
                with st.expander(f"🔄 {interv['name']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Before", f"{analysis['risk_before']}%")
                    col2.metric("Risk After", f"{analysis['risk_after']}%",
                               delta=f"{analysis['risk_change']:+.1f}%")
                    col3.metric("Level Change",
                               f"{analysis['level_before']} → {analysis['level_after']}")

                    if analysis.get("indicator_changes"):
                        st.markdown("**Indicator Changes:**")
                        for change in analysis["indicator_changes"]:
                            direction = "📉" if change["direction"] == "improved" else "📈"
                            st.write(f"{direction} {change['indicator']}: "
                                    f"{change['before']} → {change['after']} ({change['change_pct']:+.1f}%)")

                    st.warning(f"**Side Effects:** {', '.join(analysis['side_effects'])}")

            # Custom what-if
            st.divider()
            st.markdown("#### 🧪 Custom What-If Test")
            available = result["interventions"]["available_interventions"]
            int_names = {i["name"]: i["id"] for i in available}
            selected = st.selectbox("Select intervention:", list(int_names.keys()))
            magnitude = st.slider("Intervention magnitude:", 0.5, 2.0, 1.0, 0.1)

            if st.button("Run What-If Analysis"):
                custom_result = predictor.what_if(indicators, int_names[selected],
                                                   risk_score, magnitude)
                st.json(custom_result)

    # TAB 5: Policy
    with tabs[4]:
        st.subheader("Policy Recommendations")
        if result["policy_recommendations"]:
            policy = result["policy_recommendations"]

            st.markdown(f"**Friction Type:** {policy['friction_type'].replace('_', ' ').title()}")
            st.markdown(f"**Urgency:** {policy['urgency']}")

            for timeline, data in policy["recommendations"].items():
                st.markdown(f"### ⏰ {timeline.replace('_', ' ').title()} ({data['timeline']})")
                for action in data["actions"]:
                    priority_emoji = "🔴" if action["priority"] == "high" else "🟡" if action["priority"] == "medium" else "⚪"
                    st.markdown(f"{priority_emoji} **{action['action']}** | "
                               f"Cost: {action['cost']} | Impact: {action['expected_impact']}")

            if policy.get("historical_lessons"):
                st.markdown("### 📚 Historical Lessons")
                for lesson in policy["historical_lessons"]:
                    st.info(lesson["lesson"])

    # TAB 6: Full Report
    with tabs[5]:
        st.subheader("Full JSON Report")
        st.json(result)

        if st.button("💾 Download Report"):
            report_json = json.dumps(result, indent=2, default=str)
            st.download_button("Download JSON", report_json,
                              "dsfs_report.json", "application/json")

else:
    # Landing page
    st.markdown("### 👈 Adjust indicators in the sidebar, then click **PREDICT RISK**")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🤖 6-Layer ML Pipeline")
        st.write("LGBM + CNN-LSTM + SBERT + GCRI + What-If + Policy Engine")
    with col2:
        st.markdown(f"#### 📚 {len(HISTORICAL_CASES)} Historical Cases")
        st.write("Real-world conflict data from 2010-2024")
    with col3:
        st.markdown("#### 🔮 Multi-Horizon Forecast")
        st.write("Predicts risk at 1-month, 3-month, and 6-month horizons")

    # Quick presets
    st.markdown("### 🎯 Quick Presets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**India 2024**")
        st.code("Unemployment: 8%\nInflation: 5%\nGini: 0.35\nYouth: 52%")
    with col2:
        st.markdown("**Arab Spring (2011)**")
        st.code("Unemployment: 13%\nInflation: 11%\nGini: 0.36\nYouth: 60%")
    with col3:
        st.markdown("**Sri Lanka Crisis (2022)**")
        st.code("Unemployment: 5%\nInflation: 55%\nGini: 0.39\nFood: 180")
