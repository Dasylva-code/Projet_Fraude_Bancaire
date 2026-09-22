"""Application Streamlit — détection de fraude sur cartes bancaires.

Lancement : streamlit run app.py   (après avoir exécuté train.py)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from features import RAW_FEATURES  # noqa: F401  (l'import garantit que add_hour est résolu au chargement)

st.set_page_config(page_title="Détection de fraude bancaire", page_icon="💳", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("modele_pipeline.pkl")


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv("data/sample_test.csv")


@st.cache_resource
def get_explainer(_clf):
    return shap.TreeExplainer(_clf)


def explain(pipe, row: pd.DataFrame) -> pd.Series:
    """Contributions SHAP (en log-odds) de chaque variable pour une transaction."""
    Xt = pipe.named_steps["prep"].transform(pipe.named_steps["fe"].transform(row))
    Xt.columns = [c.split("__")[-1] for c in Xt.columns]
    values = get_explainer(pipe.named_steps["clf"]).shap_values(Xt)
    return pd.Series(values[0], index=Xt.columns)


def business_cost(y_true, flagged, amounts, fp_cost):
    missed = amounts[(y_true == 1) & (~flagged)].sum()
    false_alerts = ((y_true == 0) & flagged).sum()
    return float(missed + fp_cost * false_alerts)


if not Path("modele_pipeline.pkl").exists():
    st.error("Modèle introuvable : lance d'abord `python train.py`.")
    st.stop()

bundle = load_bundle()
pipe = bundle["pipeline"]

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("Paramètres de décision")
threshold = st.sidebar.slider(
    "Seuil de blocage", 0.0, 1.0, float(bundle["threshold"]), 0.005,
    help="Une transaction est bloquée si sa probabilité de fraude dépasse ce seuil. "
         "Valeur par défaut : seuil qui minimise le coût métier à l'entraînement.",
)
fp_cost = st.sidebar.number_input(
    "Coût d'une fausse alerte (€)", min_value=0.0, value=float(bundle["fp_cost"]), step=1.0,
    help="Revue manuelle, friction client... Hypothèse illustrative.",
)
st.sidebar.caption(
    "⚠️ Projet pédagogique / expérimental. Ce modèle n'est pas un outil de décision "
    "réglementaire et n'a pas été validé pour un usage bancaire réel."
)

st.title("💳 Détection de fraude sur cartes bancaires")
tab_sim, tab_csv, tab_perf = st.tabs(["Simuler une transaction", "Analyser un fichier CSV", "Performances du modèle"])

# ---------------------------------------------------------------- Tab 1
with tab_sim:
    sample = load_sample()
    st.caption(
        "Les variables V1 à V28 sont anonymisées (PCA) : on choisit donc une transaction du jeu de test, "
        "puis on peut modifier le montant et l'heure. Cet échantillon contient toutes les fraudes du test "
        "et seulement 300 transactions légitimes : le taux de fraude y est bien plus élevé qu'en réalité."
    )
    idx = st.selectbox(
        "Transaction du jeu de test", sample.index,
        format_func=lambda i: f"#{i} — {sample.loc[i, 'Amount']:.2f} €",
    )
    orig = sample.loc[idx]
    orig_hour = int((orig["Time"] // 3600) % 24)

    c1, c2 = st.columns(2)
    amount = c1.number_input("Montant (€)", min_value=0.0, value=float(orig["Amount"]), step=1.0)
    hour = c2.slider("Heure de la transaction", 0, 23, orig_hour)

    row = sample.loc[[idx], RAW_FEATURES].copy()
    row["Amount"] = amount
    row["Time"] = row["Time"] - orig_hour * 3600 + hour * 3600

    proba = float(pipe.predict_proba(row)[0, 1])
    blocked = proba >= threshold

    m1, m2 = st.columns(2)
    m1.metric("Probabilité de fraude", f"{proba:.1%}")
    m2.metric("Seuil de blocage", f"{threshold:.1%}")
    if blocked:
        st.error("🚫 Transaction **bloquée** (suspecte)")
    else:
        st.success("✅ Transaction **autorisée**")

    st.subheader("Pourquoi ce score ?")
    contrib = explain(pipe, row)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#1f77b4" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Contribution SHAP (rouge = pousse vers « fraude »)")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Vérité terrain (jeu de test)"):
        st.write("Fraude réelle" if int(orig["Class"]) == 1 else "Transaction légitime")

# ---------------------------------------------------------------- Tab 2
with tab_csv:
    st.write("Colonnes attendues : `Time`, `V1`…`V28`, `Amount` (et optionnellement `Class` pour évaluer).")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in RAW_FEATURES if c not in data.columns]
        if missing:
            st.error(f"Colonnes manquantes : {', '.join(missing)}")
        else:
            scored = data.copy()
            scored["proba_fraude"] = pipe.predict_proba(data[RAW_FEATURES])[:, 1]
            scored["bloquee"] = scored["proba_fraude"] >= threshold

            k1, k2 = st.columns(2)
            k1.metric("Transactions analysées", f"{len(scored):,}".replace(",", " "))
            k2.metric("Transactions bloquées", f"{int(scored['bloquee'].sum()):,}".replace(",", " "))

            if "Class" in scored.columns:
                y, flagged, amounts = scored["Class"].to_numpy(), scored["bloquee"].to_numpy(), scored["Amount"].to_numpy()
                tp = int(((y == 1) & flagged).sum()); fp = int(((y == 0) & flagged).sum())
                fn = int(((y == 1) & ~flagged).sum()); tn = int(((y == 0) & ~flagged).sum())
                st.write(pd.DataFrame([[tn, fp], [fn, tp]], index=["Légitime", "Fraude"],
                                      columns=["Autorisée", "Bloquée"]))
                r1, r2, r3 = st.columns(3)
                r1.metric("Recall", f"{tp / max(tp + fn, 1):.1%}")
                r2.metric("Precision", f"{tp / max(tp + fp, 1):.1%}")
                r3.metric("Coût métier", f"{business_cost(y, flagged, amounts, fp_cost):,.0f} €".replace(",", " "),
                          help="Montants des fraudes non détectées + coût des fausses alertes")

            st.dataframe(scored.sort_values("proba_fraude", ascending=False).head(100))
            st.download_button("Télécharger les résultats (CSV)", scored.to_csv(index=False).encode(),
                               "resultats_fraude.csv", "text/csv")

# ---------------------------------------------------------------- Tab 3
with tab_perf:
    mfile = Path("reports/metrics.json")
    if mfile.exists():
        m = json.loads(mfile.read_text())
        t = m["test"]
        st.write(f"Modèle retenu : **{m['model']}** — seuil optimal à l'entraînement : **{m['threshold']:.3f}**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("PR-AUC (test)", f"{t['pr_auc']:.3f}")
        p2.metric("Recall", f"{t['recall']:.1%}")
        p3.metric("Precision", f"{t['precision']:.1%}")
        p4.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
        st.write(f"Coût métier sur le test : **{t['cost_model']:,.0f} €** contre **{t['cost_no_model']:,.0f} €** "
                 "sans modèle (toutes les fraudes passent).".replace(",", " "))
        cols = st.columns(2)
        for i, name in enumerate(["pr_curve", "confusion_matrix", "cost_vs_threshold", "shap_summary"]):
            img = Path(f"reports/{name}.png")
            if img.exists():
                cols[i % 2].image(str(img))
    else:
        st.info("Lance `python train.py` pour générer les métriques.")
