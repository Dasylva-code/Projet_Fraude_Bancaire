# 💳 Détection de fraude sur cartes bancaires

> ⚠️ **Projet pédagogique / expérimental.** Ce modèle n'est pas un outil de décision réglementaire et n'a pas été validé pour un usage bancaire réel.

<!-- Ajoute ici une capture d'écran de l'application : ![App](reports/capture_app.png) -->

## 1. Problème métier
Une banque doit identifier les transactions suspectes **avant** leur validation finale. Deux erreurs coûtent de l'argent :
- une **fraude non détectée** coûte le montant de la transaction ;
- une **fausse alerte** coûte une revue manuelle et de la friction client (hypothèse : 5 € par alerte, paramétrable).

Le projet vise donc à **minimiser un coût métier**, pas à maximiser une accuracy.

## 2. Données
- **Credit Card Fraud Detection** (Kaggle, ULB) : transactions de porteurs de cartes européens sur 2 jours.
- Variables : `Time`, `Amount`, `V1`–`V28` (composantes PCA anonymisées), `Class` (1 = fraude).
- Très fort déséquilibre : ≈ 0,17 % de fraudes. *(à confirmer avec tes chiffres après suppression des doublons)*
- Voir `data/README.md` pour le téléchargement.

## 3. Méthodologie (protégée contre le Data Leakage)
1. **EDA** (`notebooks/EDA.ipynb`) : qualité, doublons, déséquilibre, montant/heure, variables discriminantes.
2. **Doublons supprimés avant le split**, puis **split train/test stratifié** isolé dès le départ.
3. **Un seul pipeline `imblearn`** (`features.py` + `train.py`) : heure de la journée → `RobustScaler` sur `Amount` → (SMOTE) → XGBoost.
   SMOTE est *dans* le pipeline : il n'est appliqué qu'aux données d'entraînement de chaque fold de validation croisée.
4. **Comparaison de 3 stratégies** en validation croisée stratifiée (5 folds), sur le train uniquement, avec la **PR-AUC** :
   régression logistique (baseline), XGBoost pondéré (`scale_pos_weight`), XGBoost + SMOTE.
5. **Choix du seuil de décision** sur les probabilités *out-of-fold* du train (jamais sur le test), en minimisant le coût métier.
6. **Évaluation finale unique** sur le test : PR-AUC, Recall, Precision, matrice de confusion, coût vs « sans modèle ».
7. **Explicabilité** : SHAP (globale dans `reports/`, locale dans l'application).

## 4. Résultats
*(à compléter avec les valeurs de `reports/metrics.json` après avoir lancé `train.py` sur le vrai dataset)*

| Modèle | PR-AUC (CV, out-of-fold) |
|---|---|
| Régression logistique (baseline) | … |
| XGBoost + pondération des classes | … |
| XGBoost + SMOTE | … |

Sur le jeu de test, au seuil choisi : PR-AUC = …, Recall = …, Precision = …, coût métier = … € (contre … € sans modèle).

![Courbe PR](reports/pr_curve.png)
![Matrice de confusion](reports/confusion_matrix.png)
![SHAP](reports/shap_summary.png)

## 5. Limites
- Variables V1–V28 anonymisées : pas d'interprétation métier directe des facteurs.
- Données de 2013, 2 jours seulement : pas de test de robustesse à la dérive temporelle (*concept drift*), or les schémas de fraude évoluent vite.
- Le coût d'une fausse alerte (5 €) est une **hypothèse** ; le seuil optimal en dépend.
- Peu de fraudes (quelques centaines) : les métriques ont une forte variance.
- Pas de validation réglementaire, pas de monitoring en production.

## 6. Lancer le projet
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# placer creditcard.csv dans data/ (voir data/README.md)
python train.py            # entraîne, évalue, sauvegarde modele_pipeline.pkl + reports/
streamlit run app.py       # ouvre l'application
```

## 7. Structure
```
├── data/                 # données (creditcard.csv à télécharger) + sample_test.csv pour l'app
├── notebooks/EDA.ipynb   # exploration
├── reports/              # métriques et figures générées par train.py
├── features.py           # feature engineering partagé (train + app)
├── train.py              # pipeline, validation, seuil, évaluation, sauvegarde
├── app.py                # application Streamlit
├── modele_pipeline.pkl   # pipeline + seuil sauvegardés (joblib)
└── requirements.txt
```
