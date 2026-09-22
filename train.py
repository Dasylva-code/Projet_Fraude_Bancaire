"""Entraînement du pipeline de détection de fraude — protégé contre le Data Leakage.

Démarche :
1. Chargement, suppression des doublons AVANT le split.
2. Split train/test stratifié isolé dès le début (le test n'est utilisé qu'une fois).
3. Comparaison de stratégies par validation croisée stratifiée sur le train uniquement
   (SMOTE est dans le pipeline imblearn => appliqué seulement sur les folds d'entraînement).
4. Choix du seuil de décision sur les probabilités out-of-fold (jamais sur le test),
   en minimisant un coût métier.
5. Évaluation finale unique sur le test + sauvegarde du modèle et des figures.

Usage : python train.py [--data data/creditcard.csv] [--fp-cost 5.0]
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # et non sklearn.pipeline : SMOTE est un "sampler"
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from xgboost import XGBClassifier

from features import RAW_FEATURES, add_hour

SEED = 42
REPORTS = Path("reports")


def build_pipeline(kind: str, scale_pos_weight: float = 1.0) -> Pipeline:
    """Pipeline complet : entrée = colonnes brutes (Time, V1..V28, Amount)."""
    fe = FunctionTransformer(add_hour)
    # Seul Amount (très asymétrique) est mis à l'échelle : V1..V28 sont déjà des composantes PCA.
    prep = ColumnTransformer([("amount", RobustScaler(), ["Amount"])], remainder="passthrough")
    prep.set_output(transform="pandas")

    xgb_params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
        random_state=SEED, n_jobs=-1,
    )
    if kind == "logreg":  # baseline
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        return Pipeline([("fe", fe), ("prep", prep), ("clf", clf)])
    if kind == "xgb_weight":  # pondération des classes, sans rééchantillonnage
        clf = XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_params)
        return Pipeline([("fe", fe), ("prep", prep), ("clf", clf)])
    if kind == "xgb_smote":  # SMOTE appliqué uniquement sur les données d'entraînement de chaque fold
        clf = XGBClassifier(**xgb_params)
        return Pipeline([("fe", fe), ("prep", prep), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    raise ValueError(kind)


def business_cost(y_true, proba, amounts, threshold, fp_cost):
    """Coût = montants des fraudes non détectées + coût fixe par fausse alerte."""
    y_pred = proba >= threshold
    missed = amounts[(y_true == 1) & (~y_pred)].sum()
    false_alerts = ((y_true == 0) & y_pred).sum()
    return float(missed + fp_cost * false_alerts)

# remplacement
def best_threshold(y_true, proba, amounts, fp_cost, tol=0.01):
    grid = np.linspace(0.005, 0.995, 199)
    costs = np.array([business_cost(y_true, proba, amounts, t, fp_cost) for t in grid])
    ok = grid[costs <= costs.min() * (1 + tol)]   # zone du plateau
    return float(np.median(ok)), grid, costs


def main(data_path: str, fp_cost: float) -> None:
    REPORTS.mkdir(exist_ok=True)

    # 1. Chargement + doublons supprimés avant le split (sinon une même ligne peut être en train ET en test)
    df = pd.read_csv(data_path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Doublons supprimés : {n_before - len(df)} | fraudes : {df['Class'].sum()} ({df['Class'].mean():.3%})")

    X, y = df[RAW_FEATURES], df["Class"]

    # 2. Split isolé dès le début
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())

    # 3. Comparaison par CV stratifiée sur le TRAIN uniquement (probabilités out-of-fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = ["logreg", "xgb_weight", "xgb_smote"]
    oof, cv_pr_auc = {}, {}
    for name in candidates:
        print(f"CV : {name} ...")
        oof[name] = cross_val_predict(
            build_pipeline(name, ratio), X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        cv_pr_auc[name] = float(average_precision_score(y_train, oof[name]))
        print(f"   PR-AUC (OOF) = {cv_pr_auc[name]:.4f}")

    # Le modèle retenu est le meilleur des variantes XGBoost (SHAP nécessite un modèle à base d'arbres) ;
    # la régression logistique sert de baseline de comparaison.
    best = max(["xgb_weight", "xgb_smote"], key=lambda k: cv_pr_auc[k])
    print(f"Modèle retenu : {best}")

    # 4. Seuil choisi sur les probabilités OOF du train (pas sur le test)
    amounts_train = X_train["Amount"].to_numpy()
    threshold, grid, costs = best_threshold(y_train.to_numpy(), oof[best], amounts_train, fp_cost)
    print(f"Seuil optimal (coût métier, fp_cost={fp_cost}) : {threshold:.2f}")

    # 5. Réentraînement sur tout le train, évaluation unique sur le test
    pipe = build_pipeline(best, ratio).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    amounts_test = X_test["Amount"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    baseline = build_pipeline("logreg").fit(X_train, y_train)
    base_proba = baseline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": best,
        "fp_cost": fp_cost,
        "threshold": threshold,
        "cv_pr_auc_oof": cv_pr_auc,
        "test": {
            "pr_auc": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "recall": float(recall_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "baseline_logreg_pr_auc": float(average_precision_score(y_test, base_proba)),
            "cost_model": business_cost(y_test.to_numpy(), proba, amounts_test, threshold, fp_cost),
            "cost_no_model": float(amounts_test[y_test.to_numpy() == 1].sum()),
        },
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=2))

    # Figures
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=f"{best}", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, base_proba, name="baseline logreg", ax=ax)
    ax.set_title("Courbe Précision-Rappel (jeu de test)")
    fig.savefig(REPORTS / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Matrice de confusion (seuil = {threshold:.2f})")
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, costs)
    ax.axvline(threshold, color="red", ls="--", label=f"optimum = {threshold:.2f}")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Coût métier (train, out-of-fold)")
    ax.set_title("Choix du seuil")
    ax.legend()
    fig.savefig(REPORTS / "cost_vs_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Explicabilité globale (SHAP) sur un échantillon du test
    try:
        import shap
    # remplacement
        sample = pd.concat([X_test[y_test == 1], X_test[y_test == 0].sample(1000, random_state=SEED)])
        Xt = pipe.named_steps["prep"].transform(pipe.named_steps["fe"].transform(sample))
        Xt.columns = [c.split("__")[-1] for c in Xt.columns]
        shap_values = shap.TreeExplainer(pipe.named_steps["clf"])(Xt)
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # SHAP est un bonus : ne bloque pas l'entraînement
        print(f"SHAP ignoré : {exc}")

    # Sauvegarde : pipeline + seuil dans un seul fichier
    joblib.dump(
        {"pipeline": pipe, "threshold": threshold, "fp_cost": fp_cost, "model_name": best},
        "modele_pipeline.pkl",
    )

    # Échantillon de démo pour l'app (le CSV complet dépasse la limite GitHub de 100 Mo)
    demo = pd.concat([
        X_test[y_test == 1].assign(Class=1),
        X_test[y_test == 0].sample(300, random_state=SEED).assign(Class=0),
    ]).sample(frac=1, random_state=SEED)
    demo.to_csv("data/sample_test.csv", index=False)
    print("Sauvegardé : modele_pipeline.pkl, reports/, data/sample_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--fp-cost", type=float, default=5.0,
                        help="Coût estimé d'une fausse alerte (revue manuelle / friction client), en €")
    args = parser.parse_args()
    main(args.data, args.fp_cost)
