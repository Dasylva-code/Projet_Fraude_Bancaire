## Résultats

| Modèle | PR-AUC (CV, out-of-fold) |
|---|---|
| Régression logistique (baseline) | 0,742 |
| XGBoost + pondération des classes | **0,850** |
| XGBoost + SMOTE | 0,842 |

Sur le jeu de test (283 726 transactions, 95 fraudes), au seuil retenu (0,51, centre du plateau de coût
sur la validation croisée) : PR-AUC = 0,819, Recall = 78,9 %, Precision = 93,75 %,
coût métier = 3 814 € contre 14 766 € sans modèle (hypothèse : 5 € par fausse alerte).

75 fraudes sur 95 sont détectées, pour seulement 5 fausses alertes sur 56 651 transactions légitimes.
Les 20 fraudes manquées concentrent l'essentiel du coût restant.

- Avec seulement 95 fraudes dans le test, le Recall (78,9 %) varie sensiblement selon le split :
  une fraude de plus ou de moins déplace le chiffre d'environ 1 point.
- Le seuil de décision est choisi sur le centre d'un plateau de coût quasi plat entre 0,1 et 0,78,
  plutôt qu'un minimum ponctuel plus instable — mais le coût final ne dépend presque pas de ce choix
  précis dans cette zone.
- Les variables V1–V28 sont anonymisées (PCA) : les 5 facteurs les plus influents selon SHAP
  (V14, V4, V12, V10, V11) confirment l'écart de distribution observé en EDA, mais sans
  interprétation métier possible.
- Split aléatoire (pas temporel) sur 2 jours de données de 2013 : pas de test de robustesse
  face à l'évolution des schémas de fraude dans le temps.

![Application — transaction bloquée](reports/app_transaction_bloquee.png)

### Exemple de faux négatif

Le modèle n'est pas infaillible : sur la transaction ci-dessous (519,90 €), il estime la probabilité
de fraude à seulement 0,1 % et l'autorise, alors qu'il s'agit d'une fraude réelle. C'est l'un des
20 faux négatifs du jeu de test.

![Application — faux négatif](reports/app_transaction_autorisee_fn.png)

![Application — import CSV](reports/app_csv_complet.png)

*(les métriques affichées ici portent sur le fichier complet, entraînement inclus, donc elles sont
optimistes — voir la section Résultats ci-dessus pour la mesure fiable, sur le jeu de test seul)*

![Application — onglet Performances](reports/app_performances.png)