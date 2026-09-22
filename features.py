"""Feature engineering partagé entre train.py et app.py.

Ce module doit rester importable au moment du chargement du pipeline sauvegardé
(joblib retrouve la fonction `add_hour` par son nom de module).
"""
import pandas as pd

RAW_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def add_hour(X: pd.DataFrame) -> pd.DataFrame:
    """Remplace `Time` (secondes écoulées depuis la 1re transaction, sur 2 jours)
    par l'heure de la journée (0-23), plus interprétable et sans dérive temporelle."""
    X = X.copy()
    X["Hour"] = (X["Time"] // 3600) % 24
    return X.drop(columns=["Time"])
