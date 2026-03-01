import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend import (
    compute_similarity,
    risk_analysis,
    budget_score,
    collaboration_score,
    success_probability
)

from database import startups

st.set_page_config(page_title="AI Startup–Industry Matchmaker", layout="wide")

st.title("🚀 AI Startup–Industry Collaboration Matchmaker")
st.markdown("### Intelligent Semantic Compatibility & Risk-Based Ranking System")

# -------- Industry Input --------
st.header("🏢 Industry Requirements")

industry_problem = st.text_area("Describe Industry Problem Statement")
industry_tech = st.text_input("Required Technologies (comma separated)")
industry_budget = st.selectbox("Budget Range", ["Low", "Medium", "High"])

if st.button("🔍 Find Best Startup Matches"):

    if industry_problem and industry_tech:

        industry_text = industry_problem + " " + industry_tech

        results = []

        for startup in startups:

            startup_text = startup["domain"] + " " + startup["tech"]

            match_percentage = compute_similarity(industry_text, startup_text)
            risk_score = risk_analysis(startup["stage"])
            budget_adjustment = budget_score(industry_budget, startup["stage"])

            final_score = collaboration_score(
                match_percentage,
                risk_score,
                budget_adjustment
            )

            success_rate = success_probability(final_score)

            results.append({
                "Startup": startup["name"],
                "Match %": match_percentage,
                "Risk Score": risk_score,
                "Budget Adj": budget_adjustment,
                "Final Collaboration Score": final_score,
                "Success Prediction": success_rate
            })

        # Rank startups
        ranked_results = sorted(
            results,
            key=lambda x: x["Final Collaboration Score"],
            reverse=True
        )

        st.subheader("📊 Ranked Startup Recommendations")

        for idx, result in enumerate(ranked_results):

            st.markdown(f"## 🏆 Rank {idx+1}: {result['Startup']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Semantic Match %", f"{result['Match %']}%")
            col2.metric("Risk Score", result["Risk Score"])
            col3.metric("Final Score", result["Final Collaboration Score"])

            st.metric("Success Prediction", result["Success Prediction"])

            if result["Final Collaboration Score"] > 70:
                st.success("🔥 Highly Recommended for Immediate Collaboration")
            elif result["Final Collaboration Score"] > 40:
                st.warning("⚡ Moderate Potential – Further Evaluation Needed")
            else:
                st.error("❌ Low Alignment")

            st.markdown("---")

        # -------- Graph Visualization --------
        df = pd.DataFrame(ranked_results)

        st.subheader("📊 Collaboration Score Comparison")

        fig, ax = plt.subplots()
        ax.bar(df["Startup"], df["Final Collaboration Score"])
        ax.set_ylabel("Final Collaboration Score")
        ax.set_xlabel("Startup")
        plt.xticks(rotation=30)

        st.pyplot(fig)

        # -------- Explainability --------
        st.subheader("🔎 Why This Match?")

        st.write("""
        • Semantic similarity is computed using Sentence Transformers.\n
        • Risk score is based on startup maturity stage.\n
        • Budget compatibility adjusts final recommendation.\n
        • Final score combines compatibility, risk, and budget alignment.\n
        """)

    else:
        st.warning("Please fill Industry Problem and Technologies.")
