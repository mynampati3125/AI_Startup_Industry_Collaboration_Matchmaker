from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load semantic model once
model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_similarity(industry_text, startup_text):
    industry_text = industry_text.lower()
    startup_text = startup_text.lower()

    embeddings = model.encode([industry_text, startup_text])
    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(score * 100, 2)


def risk_analysis(startup_stage):
    risk_map = {
        "Idea": 70,
        "Prototype": 50,
        "MVP": 30,
        "Scaling": 10
    }
    return risk_map.get(startup_stage, 50)


def budget_score(industry_budget, startup_stage):
    stage_cost = {
        "Idea": "Low",
        "Prototype": "Low",
        "MVP": "Medium",
        "Scaling": "High"
    }

    if industry_budget == stage_cost.get(startup_stage):
        return 10
    elif industry_budget == "High":
        return 5
    else:
        return -5


def collaboration_score(match_percentage, risk_score, budget_adjustment):
    final_score = match_percentage - (risk_score * 0.2) + budget_adjustment
    return round(final_score, 2)


def success_probability(final_score):
    if final_score > 70:
        return "High Success Probability (85%)"
    elif final_score > 40:
        return "Moderate Success Probability (60%)"
    else:
        return "Low Success Probability (35%)"
