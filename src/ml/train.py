"""Обучение и оценка моделей: RandomForest + LogisticRegression.

Валидация: GroupKFold по child_id — предотвращает утечку между train/test.
Метрики: ROC-AUC, precision, recall, F1 при пороге 0.5.

Pipeline:
    - SimpleImputer(median) — заполняет NaN (например, avg_days_between_improvements
      при < 2 улучшениях)
    - StandardScaler — только для LogReg
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


@dataclass
class CVResult:
    model_name: str
    roc_auc_mean: float
    roc_auc_std: float
    precision_mean: float
    recall_mean: float
    f1_mean: float
    fold_metrics: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "model": self.model_name,
            "roc_auc": round(self.roc_auc_mean, 4),
            "roc_auc_std": round(self.roc_auc_std, 4),
            "precision": round(self.precision_mean, 4),
            "recall": round(self.recall_mean, 4),
            "f1": round(self.f1_mean, 4),
        }


def _make_rf() -> Pipeline:
    """RF с imputer. Imputer нужен, чтобы не падать на NaN и быть
    консистентным с LogReg по предобработке."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _make_logreg() -> Pipeline:
    """Imputer → Scaler → LogReg. Порядок важен: scaler не переносит NaN."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def _n_splits(groups: pd.Series) -> int:
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(f"Нужно минимум 2 уникальных child_id, а их {n_groups}.")
    return min(5, n_groups)


def cross_validate_model(
    model,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> CVResult:
    """GroupKFold CV. Считает метрики по фолдам."""
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, groups должны быть одной длины")

    gkf = GroupKFold(n_splits=_n_splits(groups))

    aucs, precisions, recalls, f1s = [], [], [], []
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if y_tr.nunique() < 2:
            continue

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        # ROC-AUC определён только при 2 классах в y_te
        if y_te.nunique() >= 2:
            try:
                auc = roc_auc_score(y_te, proba)
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")

        pred = (proba >= 0.5).astype(int)
        precision = precision_score(y_te, pred, zero_division=0)
        recall = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)

        aucs.append(auc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fold_metrics.append({
            "fold": fold_i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    if not fold_metrics:
        raise ValueError("Не удалось посчитать метрики ни на одном фолде.")

    return CVResult(
        model_name=model_name,
        roc_auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        roc_auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        precision_mean=float(np.mean(precisions)),
        recall_mean=float(np.mean(recalls)),
        f1_mean=float(np.mean(f1s)),
        fold_metrics=fold_metrics,
    )


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Обучает RF и LogReg с CV, возвращает таблицу метрик и fitted-модели."""
    results = []
    fitted = {}

    for model, name in [
        (_make_rf(), "RandomForest"),
        (_make_logreg(), "LogisticRegression"),
    ]:
        cv = cross_validate_model(model, name, X, y, groups)
        results.append(cv.as_row())

        # финальное обучение на всех данных (для SHAP и инференса)
        model.fit(X, y)
        fitted[name] = model

    metrics_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    return metrics_df, fitted


def fit_final(model_name: str, X: pd.DataFrame, y: pd.Series) -> object:
    """Обучает одну модель на всех данных."""
    model = _make_rf() if model_name == "RandomForest" else _make_logreg()
    model.fit(X, y)
    return model