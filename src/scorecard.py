import numpy as np


def prob_to_score(prob, offset=500, factor=50):
    odds = (1 - prob) / max(prob, 1e-6)
    return offset + factor * np.log(odds)


def risk_grade(score):
    if score < 500:
        return "High Risk"
    elif score < 650:
        return "Medium Risk"
    elif score < 750:
        return "Low Risk"
    else:
        return "Very Low Risk"
