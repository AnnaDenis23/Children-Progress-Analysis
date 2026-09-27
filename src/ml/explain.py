"""SHAP-объяснения для RandomForest (в том числе внутри sklearn Pipeline).

Pipeline распаковывается:
1. `_unwrap_model` достаёт финальный `clf` (RandomForest) из Pipeline.
2. `_unwrap_X` прогоняет признаки через шаги до `clf` (imputer),
   чтобы SHAP работал с той же матрицей, что видит модель.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Распаковка Pipeline
# ---------------------------------------------------------------------------
def _unwrap_model(model):
    """Достаёт финальный estimator из sklearn Pipeline."""
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return model.named_steps["clf"]
    return model


def _unwrap_X(model, X: pd.DataFrame) -> pd.DataFrame:
    """Прогоняет X через шаги Pipeline до `clf` включительно (не включая).

    Возвращает DataFrame с теми же колонками, что и исходный X.
    Это важно, чтобы SHAP показал имена признаков, а не индексы.
    """
    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return X

    Xt = X
    for step_name, step in model.named_steps.items():
        if step_name == "clf":
            break
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)

    if not isinstance(Xt, pd.DataFrame):
        Xt = pd.DataFrame(Xt, columns=X.columns, index=X.index)

    return Xt


def _normalize_shap_values(shap_values) -> np.ndarray:
    """SHAP может вернуть list[array] (бинарная классификация) или array."""
    if isinstance(shap_values, list):
        # [class_0, class_1] -> берём class_1
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # shape (n, features, 2)
        return arr[:, :, 1]
    return arr


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------
def explain_patient(
    model,
    X_row: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Топ-N признаков по |SHAP| для одного пациента.

    Parameters
    ----------
    model : fitted estimator или Pipeline
    X_row : pd.DataFrame
        Одна строка с признаками (той же структуры, что и обучающая X).
    top_n : int

    Returns
    -------
    pd.DataFrame
        Колонки: feature, value, shap_value, direction.
    """
    if not isinstance(X_row, pd.DataFrame):
        raise TypeError("X_row must be a pandas DataFrame")
    if len(X_row) != 1:
        raise ValueError("X_row должен содержать ровно одну строку")

    clf = _unwrap_model(model)
    X_proc = _unwrap_X(model, X_row)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_proc)
    sv = _normalize_shap_values(shap_values)[0]  # (n_features,)

    df = pd.DataFrame({
        "feature": X_proc.columns,
        "value": X_proc.iloc[0].values,
        "shap_value": sv,
    })

    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    df["direction"] = np.where(
        df["shap_value"] > 0,
        "повышает риск",
        "понижает риск",
    )
    df = df.drop(columns=["abs_shap"])

    return df


def global_feature_importance(model, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Средняя |SHAP| по всем примерам — глобальная важность признаков."""
    clf = _unwrap_model(model)
    X_proc = _unwrap_X(model, X)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_proc)
    sv = _normalize_shap_values(shap_values)

    importance = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({
        "feature": X_proc.columns,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    return df


def predict_patient(
    model,
    X_row: pd.DataFrame,
) -> float:
    """Вероятность класса 1 (застой) для одной строки признаков.

    Работает и с Pipeline, и с голым estimator — imputer внутри Pipeline
    сам обработает возможные NaN.
    """
    if len(X_row) != 1:
        raise ValueError("X_row должен содержать ровно одну строку")
    return float(model.predict_proba(X_row)[:, 1][0])