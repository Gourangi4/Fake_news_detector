"""Train, evaluate and save the fake news classifier.

Run from the repo root:  python -m src.train
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

from src.models import DEPLOYED_MODEL, RANDOM_STATE, TEST_SIZE, build_models
from src.preprocess import load_raw, prepare

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
for d in (MODELS, FIGS):
    d.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    raw = load_raw()
    print("raw rows:", len(raw))
    df = prepare(raw)
    print("after cleaning/dedup:", len(df), "| fake:", int((df.label == 0).sum()),
          "real:", int((df.label == 1).sum()), f"({time.time()-t0:.0f}s)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"])

    models = build_models()

    results, fitted = {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
        fitted[name] = pipe
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    best_name = max(results, key=lambda k: results[k]["f1"])
    print("best by F1:", best_name, "| deployed in app:", DEPLOYED_MODEL)

    # 5-fold CV on the training set (vectorizer refit inside each fold, no leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_models()[DEPLOYED_MODEL],
                                X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"{DEPLOYED_MODEL} 5-fold CV F1:", np.round(cv_scores, 4), "mean", round(cv_scores.mean(), 4))

    # Confusion matrix for each model, plus a comparison bar chart
    for name, pipe in fitted.items():
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pipe.predict(X_test), display_labels=["Fake", "Real"], cmap="Blues", ax=ax)
        ax.set_title(f"{name}")
        fig.tight_layout()
        fig.savefig(FIGS / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["accuracy", "precision", "recall", "f1"]
    w = 0.25
    for i, name in enumerate(results):
        ax.bar(np.arange(4) + i * w, [results[name][m] for m in metrics], w, label=name)
    ax.set_xticks(np.arange(4) + w)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.9, 1.005)
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Top words from logistic regression
    lr = fitted[DEPLOYED_MODEL]
    names = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    coef = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    top_fake = [(names[i], float(coef[i])) for i in order[:15]]
    top_real = [(names[i], float(coef[i])) for i in order[::-1][:15]]
    print("top FAKE words:", [w for w, _ in top_fake])
    print("top REAL words:", [w for w, _ in top_real])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].barh([w for w, _ in top_fake][::-1], [-c for _, c in top_fake][::-1], color="#c0392b")
    axes[0].set_title("Top words pushing toward FAKE")
    axes[1].barh([w for w, _ in top_real][::-1], [c for _, c in top_real][::-1], color="#27ae60")
    axes[1].set_title("Top words pushing toward REAL")
    fig.tight_layout()
    fig.savefig(FIGS / "top_words.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_test, lr.predict(X_test),
                                   target_names=["Fake", "Real"], output_dict=True)
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({
            "rows_after_cleaning": int(len(df)),
            "train_size": int(len(X_train)), "test_size": int(len(X_test)),
            "results": results, "best_model_by_f1": best_name, "deployed_model": DEPLOYED_MODEL,
            "sklearn_version": sklearn.__version__,
            "cv_f1_mean": float(cv_scores.mean()), "cv_f1_std": float(cv_scores.std()),
            "classification_report_deployed": report,
            "top_fake_words": top_fake, "top_real_words": top_real,
        }, f, indent=2)

    # The app explains predictions with coefficients, so ship the Logistic Regression pipeline.
    joblib.dump(lr, MODELS / "fake_news_pipeline.joblib", compress=3)
    print("saved model. total time", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
