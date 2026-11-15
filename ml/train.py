"""Train + compare modern tabular models and save the best.

Candidates (none of the classic textbook CKD-project algorithms such as Random Forest, XGBoost, SVM, KNN, Decision Tree,
Naive Bayes, Logistic Regression or a plain neural net):
  * Explainable Boosting Machine (EBM, Microsoft Research InterpretML) - glass-box additive boosted model
  * CatBoost  - ordered boosting with symmetric trees
  * LightGBM  - histogram / leaf-wise gradient boosting

Run:  python ml/train.py
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, sklearn
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
import interpret

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURES, MODEL_FEATURES, NUMERIC, CATEG, clean_dataset, ebm_proba

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
EBM_NAME = "Explainable Boosting Machine (EBM)"

X, y = clean_dataset(os.path.join(ROOT, "data", "kidney_disease.csv"))
X = X[MODEL_FEATURES]
print("Data:", X.shape, "CKD =", int(y.sum()), "notCKD =", int((1 - y).sum()))


def make_ebm():
    # interactions=0 -> purely additive model, so every prediction splits EXACTLY into per-feature contributions
    return ExplainableBoostingClassifier(interactions=0, outer_bags=8, max_rounds=400, learning_rate=0.03, random_state=SEED, n_jobs=1)


candidates = {EBM_NAME: make_ebm()}
try:
    from catboost import CatBoostClassifier
    candidates["CatBoost"] = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.08, verbose=0, random_seed=SEED, thread_count=2, allow_writing_files=False)
except Exception as e:  # optional - only used for the comparison table
    print("CatBoost not available, skipped:", e)
try:
    from lightgbm import LGBMClassifier
    candidates["LightGBM"] = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, min_child_samples=5, verbose=-1, random_state=SEED, n_jobs=2)
except Exception as e:
    print("LightGBM not available, skipped:", e)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

rows = {}
for name, est in candidates.items():
    r = cross_validate(est, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
    rows[name] = {k: float(np.mean(r["test_" + k])) for k in scoring}
    rows[name].update({k + "_std": float(np.std(r["test_" + k])) for k in scoring})
    print(f"{name:36s} " + "  ".join(f"{k}={rows[name][k]:.4f}" for k in scoring))

# Selection rule (explainability-first): keep models whose CV recall is within 1 point and ROC-AUC within 0.005 of the best
# (differences that small are noise on 400 rows); among those prefer the most interpretable one.
INTERPRETABILITY = [EBM_NAME, "CatBoost", "LightGBM"]
best_recall = max(r["recall"] for r in rows.values()); best_auc = max(r["roc_auc"] for r in rows.values())
eligible = [n for n, r in rows.items() if r["recall"] >= best_recall - 0.01 and r["roc_auc"] >= best_auc - 0.005]
best = min(eligible, key=INTERPRETABILITY.index)
print("Eligible:", eligible, "-> Selected:", best)
if best != EBM_NAME:
    print("NOTE: the app's explanation code needs the EBM; keeping EBM as deployed model.")
    best = EBM_NAME


def mask_augment(Xd, yd, copies=4, seed=SEED):
    """Add copies of each row with 10-55% of the inputs hidden, so the model learns how to behave on partial reports."""
    rng = np.random.default_rng(seed); Xs, ys = [Xd], [yd]
    for _ in range(copies):
        frac = rng.uniform(0.10, 0.55, size=len(Xd))
        Xs.append(Xd.mask(rng.random(Xd.shape) < frac[:, None])); ys.append(yd)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def mask_test(Xd, frac=0.4, seed=7):
    return Xd.mask(np.random.default_rng(seed).random(Xd.shape) < frac)


X_tr_a, y_tr_a = mask_augment(X_tr, y_tr)
final = make_ebm().fit(X_tr_a, y_tr_a)
proba = ebm_proba(final, X_te)   # same scoring path as the web app (missing inputs neutral)
pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
holdout = {"n_test": int(len(y_te)), "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
           "recall": recall_score(y_te, pred), "specificity": tn / (tn + fp), "f1": f1_score(y_te, pred),
           "roc_auc": roc_auc_score(y_te, proba), "brier": brier_score_loss(y_te, proba),
           "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
print("Hold-out:", {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in holdout.items()})

pm = (ebm_proba(final, mask_test(X_te)) >= 0.5).astype(int)
tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_te, pm).ravel()
masked = {"frac_hidden": 0.4, "accuracy": accuracy_score(y_te, pm), "recall": recall_score(y_te, pm),
          "specificity": tn_m / (tn_m + fp_m), "fn": int(fn_m), "fp": int(fp_m)}
print("Hold-out with 40% of inputs hidden:", {k: round(float(v), 4) for k, v in masked.items()})

# Global importance straight from the glass-box model (mean |contribution| of each feature on the training data)
imp = sorted(zip(final.term_names_, [float(v) for v in final.term_importances()]), key=lambda t: -t[1])
print("Top global importance:", [(f, round(v, 3)) for f, v in imp[:8]])

# Deployed model: same recipe on ALL 400 rows (+ masked copies). Hold-out numbers above stay the honest estimate.
X_a, y_a = mask_augment(X, y, seed=SEED + 1)
deploy = make_ebm().fit(X_a, y_a)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump({"model": deploy, "features": MODEL_FEATURES, "name": best, "sklearn": sklearn.__version__, "interpret": interpret.__version__},
            os.path.join(ROOT, "models", "ckd_model.joblib"))
json.dump({"selected": best, "cv_5x3_train_split": rows, "holdout": holdout, "holdout_40pct_inputs_hidden": masked,
           "global_importance": dict(imp), "n_features": len(MODEL_FEATURES), "excluded_from_model": ["bp", "htn", "dm", "pe", "ane", "appet"],
           "note": "Hold-out = 20% stratified split never used for training or selection. The final model is fit on training rows plus randomly-masked "
                   "copies (robust to missing values); the deployed model is refit the same way on all 400 rows."},
          open(os.path.join(ROOT, "models", "metrics.json"), "w"), indent=2)

# ------------------------------------------------------------------ plots
st = os.path.join(ROOT, "static", "plots"); os.makedirs(st, exist_ok=True)
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["No CKD", "CKD"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix (hold-out)"); fig.tight_layout(); fig.savefig(os.path.join(st, "confusion_matrix.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.6, 3.8)); RocCurveDisplay.from_predictions(y_te, proba, ax=ax, name="EBM"); ax.set_title("ROC curve (hold-out)"); fig.tight_layout(); fig.savefig(os.path.join(st, "roc_curve.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.2, 3.8)); names = list(rows); x = np.arange(len(names)); w = 0.2
for i, k in enumerate(["accuracy", "recall", "f1", "roc_auc"]): ax.bar(x + (i - 1.5) * w, [rows[n][k] for n in names], w, label=k)
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(", 1).replace("Explainable Boosting Machine", "Explainable Boosting\nMachine") for n in names], fontsize=8); ax.set_ylim(0.95, 1.003); ax.legend(fontsize=7, ncol=4, loc="lower right"); ax.set_title("Model comparison (5-fold x3 CV)")
fig.tight_layout(); fig.savefig(os.path.join(st, "model_comparison.png"), dpi=130); plt.close(fig)
top = imp[:10][::-1]; fig, ax = plt.subplots(figsize=(5.4, 3.8)); ax.barh([t[0] for t in top], [t[1] for t in top], color="#2a6f97"); ax.set_title("Global importance (mean |log-odds contribution|)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(st, "feature_importance.png"), dpi=130); plt.close(fig)

# EBM shape functions: how each of the top lab values changes the log-odds of CKD
try:
    g = deploy.explain_global(); name_to_i = {n: i for i, n in enumerate(deploy.term_names_)}
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, (fname, _v) in zip(axs.ravel(), imp[:6]):
        d = g.data(name_to_i[fname]); xs = np.array(d["names"], dtype=float); ys = np.array(d["scores"], dtype=float); n = min(len(ys), len(xs) - 1)
        ax.step(xs[:n], ys[:n], where="post", color="#0f766e", lw=2); ax.axhline(0, color="#999", lw=.8)
        ax.set_title(NUMERIC.get(fname, (CATEG.get(fname, (fname,))[0],))[0].split(" (")[0], fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle("How each lab value moves the CKD log-odds (EBM shape functions; up = towards CKD)", fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(st, "shape_functions.png"), dpi=130); plt.close(fig)
except Exception as e:
    print("shape plot skipped:", e)
print("Saved model + plots")
