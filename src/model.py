import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier


def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    model = XGBClassifier(
        scale_pos_weight=spw,
        max_depth=5,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def _ks_statistic(y_true, y_prob):
    # KS = max separation between CDF of positives and negatives
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(model, X_test, y_test, scaler=None):
    X_input = scaler.transform(X_test) if scaler is not None else X_test
    y_prob = model.predict_proba(X_input)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ks = _ks_statistic(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    axes[1].plot(mean_pred, fraction_pos, "s-", label="Model")
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Fraction of positives")
    axes[1].set_title("Calibration Plot")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    metrics = {"AUC": round(auc, 4), "KS": round(ks, 4), "Brier": round(brier, 4)}
    return metrics
