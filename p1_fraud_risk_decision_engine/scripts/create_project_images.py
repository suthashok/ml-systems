from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch


OUT = Path("images")
OUT.mkdir(exist_ok=True)

RESULTS = Path("results")

# ------------------------
# 1. Temporal split
# ------------------------

def temporal_split():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    # base line
    ax.plot([0, 100], [0, 0], color="black")

    # Load real split data if available
    split_file = RESULTS / "temporal_split.json"
    if split_file.exists():
        with open(split_file, "r") as f:
            data = json.load(f)
        train_pct = data.get("train_pct", 65)
        title = f"Temporal Validation Strategy (Actual Data: {train_pct:.0f}% Train)"
    else:
        train_pct = 65
        title = "Temporal Validation Strategy\n(Placeholder - Run Notebook 01 to use real data)"

    ax.add_patch(Rectangle((5, -2), train_pct - 5, 4, color="#c7d2fe", ec="black"))
    ax.add_patch(Rectangle((train_pct + 2, -2), 95 - train_pct - 2, 4, color="#fecaca", ec="black"))

    ax.text((5 + train_pct) / 2, 0, "TRAINING DATA\n(Older Transactions)", ha="center", va="center", weight="bold")
    ax.text(train_pct + (100 - train_pct) / 2, 0, "VALIDATION DATA\n(Newer Transactions)", ha="center", va="center", weight="bold")

    ax.axvline(train_pct, linestyle="--", color="black")
    ax.text(train_pct, 3, "Out-of-Time Split\n(based on TransactionDT)", ha="center")

    ax.set_title(title, pad=20)

    fig.savefig(OUT / "temporal_validation_split.png", bbox_inches="tight")
    plt.close()


# ------------------------
# 2. Threshold tradeoff
# ------------------------

def threshold_plot():
    metrics_file = RESULTS / "threshold_metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        t = df["threshold"]
        precision = df["precision"]
        recall = df["recall"]
        flagged = df["flagged_rate"]
        title = "Threshold Tuning Tradeoff (Actual Data)"
    else:
        t = np.linspace(0.01, 0.99, 100)
        # Simplified illustrative curves for fraud
        precision = t ** 1.5 
        recall = 1 - t
        flagged = 1 - t * 0.8
        title = "Threshold Tuning Tradeoff\n(Placeholder - Run Notebook 06 to use real data)"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, precision, label="Precision (Fraud captured / Total flagged)", color="#ef4444", linewidth=2)
    ax.plot(t, recall, label="Recall (Fraud captured / Total fraud)", color="#3b82f6", linewidth=2)
    ax.plot(t, flagged, label="% Transactions Flagged (FPR proxy)", color="#10b981", linestyle="--", linewidth=2)

    ax.set_xlabel("Decision Threshold (Model Score)")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    
    ax.axvline(0.6, color="gray", linestyle=":", label="Selected Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(OUT / "threshold_tradeoff.png", bbox_inches="tight")
    plt.close()


# ------------------------
# 3. Feature importance
# ------------------------

def feature_importance():
    fi_file = RESULTS / "feature_importance.csv"
    if fi_file.exists():
        df = pd.read_csv(fi_file).sort_values("importance", ascending=True).tail(10)
        features = df["feature"].tolist()
        values = df["importance"].tolist()
        title = "Top Feature Importances (Actual Model)"
    else:
        features = [
            "Velocity Features (e.g. 24h count)", 
            "Entity History (e.g. prior avg amt)", 
            "Consistency Checks (mismatches)", 
            "Missingness Flags (nulls as signal)", 
            "Device/Identity Signals",
            "Raw Amount & Categories"
        ]
        features.reverse()
        values = [0.28, 0.22, 0.15, 0.12, 0.08, 0.05]
        values.reverse()
        title = "Feature Importance by Category\n(Placeholder - Run Notebook 06 to export real data)"

    fig, ax = plt.subplots(figsize=(8, 5))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color="#93c5fd")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)

    ax.set_title(title)
    ax.set_xlabel("Relative Importance")
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig(OUT / "feature_importance.png", bbox_inches="tight")
    plt.close()


# ------------------------
# 4. Decision flow
# ------------------------

def box(ax, x, y, width, text, color="white"):
    rect = FancyBboxPatch((x, y), width, 1, boxstyle="round,pad=0.1", linewidth=1, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x + width/2, y + 0.5, text, ha="center", va="center")


def decision_flow():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # boxes
    box(ax, 0, 1.5, 2, "Live Transaction", "#f8fafc")
    box(ax, 2.8, 1.5, 2.2, "Feature Generation\n(History & Velocity)", "#e0e7ff")
    box(ax, 5.8, 1.5, 2, "Model Scoring", "#fef08a")
    box(ax, 8.6, 1.5, 2, "Decision Logic\n(Thresholds)", "#fed7aa")

    box(ax, 11.5, 2.7, 1.5, "Allow", "#bbf7d0")
    box(ax, 11.5, 1.5, 1.5, "Review", "#fef08a")
    box(ax, 11.5, 0.3, 1.5, "Decline", "#fecaca")

    # arrows
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=15))

    arrow(2.1, 2, 2.7, 2)
    arrow(5.1, 2, 5.7, 2)
    arrow(7.9, 2, 8.5, 2)

    arrow(10.7, 2, 11.4, 3.2)
    arrow(10.7, 2, 11.4, 2)
    arrow(10.7, 2, 11.4, 0.8)

    ax.set_title("Production Fraud Decision Flow", pad=15)
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, 4)

    fig.savefig(OUT / "fraud_decision_flow.png", bbox_inches="tight")
    plt.close()


# ------------------------
# main
# ------------------------

if __name__ == "__main__":
    temporal_split()
    threshold_plot()
    feature_importance()
    decision_flow()

    print("Images created in /images")
``