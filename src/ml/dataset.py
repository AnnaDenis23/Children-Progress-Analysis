"""Сборка датасета для ML: X (признаки) + y (метка застоя) + группы.

Формулировка задачи
-------------------
Для каждой пары (child_id, domain) рассматриваем все точки отсечения `cut`,
начиная с `min_history` сессий. Признаки X считаются ТОЛЬКО на `group[:cut]`
(без утечки будущего). Метка y описывает, закончилась ли ВСЯ траектория
плато (>= STAGNATION_MIN_DAYS дней без улучшения к последней сессии).

Слайдинг по cut даёт аугментацию: одна пара → несколько обучающих примеров.
Группировка по child_id при кросс-валидации предотвращает утечку.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import STAGNATION_MIN_DAYS
from src.features import compute_features


def _stagnation_label(group: pd.DataFrame, min_days: int = STAGNATION_MIN_DAYS) -> int:
    """1, если траектория заканчивается плато длиной >= min_days."""
    group = group.sort_values("session_date")
    scores = group["assessment_score"].to_numpy(dtype=float)
    dates = group["session_date"].to_numpy()

    imp_idx = np.where(np.diff(scores) > 0)[0] + 1
    last_imp_date = pd.to_datetime(dates[imp_idx[-1]]) if len(imp_idx) else pd.to_datetime(dates[0])
    last_date = pd.to_datetime(dates[-1])

    return int((last_date - last_imp_date).days >= min_days)


def build_dataset(
    df: pd.DataFrame,
    min_history: int = 4,
    min_future_days: int = STAGNATION_MIN_DAYS,
) -> pd.DataFrame:
    """Собирает датасет для ML.

    Parameters
    ----------
    df : pd.DataFrame
        Очищенный df (после clean_data + fix_progress_and_specialist).
    min_history : int
        Минимальное число сессий до точки отсечения.
    min_future_days : int
        Минимальный "горизонт" будущего, чтобы метка была надёжной.

    Returns
    -------
    pd.DataFrame
        Колонки: child_id, domain, cut, y, + все числовые признаки.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = {"child_id", "domain", "session_date", "assessment_score",
                "comment", "specialist_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    df = df.copy().dropna(subset=["session_date", "assessment_score"])
    df = df.sort_values(["child_id", "domain", "session_date"])

    samples = []

    for (child_id, domain), group in df.groupby(["child_id", "domain"], sort=False):
        group = group.sort_values("session_date").reset_index(drop=True)
        n = len(group)

        if n < min_history + 2:
            continue

        y = _stagnation_label(group)

        for cut in range(min_history, n):
            history = group.iloc[:cut]
            future = group.iloc[cut:]

            if len(future) < 2:
                continue

            days_future = (future["session_date"].iloc[-1]
                           - future["session_date"].iloc[0]).days
            is_last_cut = (cut == n - 1)

            if days_future < min_future_days and not is_last_cut:
                continue

            try:
                feats = compute_features(history)
            except Exception:
                continue

            if feats.empty:
                continue

            row = feats.iloc[0].to_dict()
            row.pop("child_id", None)
            row.pop("domain", None)

            row.update({
                "child_id": child_id,
                "domain": domain,
                "cut": cut,
                "y": y,
            })
            samples.append(row)

    dataset = pd.DataFrame(samples)

    if dataset.empty:
        raise ValueError(
            "Датасет пуст: мало данных или слишком строгие min_history/min_future_days."
        )

    return dataset


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Числовые признаки + one-hot domain. Без служебных колонок и target."""
    service = {"child_id", "domain", "cut", "y"}

    numeric_cols = [
        c for c in dataset.columns
        if c not in service and pd.api.types.is_numeric_dtype(dataset[c])
    ]

    domain_dummies = pd.get_dummies(dataset["domain"], prefix="domain")
    domain_cols = list(domain_dummies.columns)

    return numeric_cols + domain_cols


def prepare_xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Возвращает (X, y, groups)."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")

    if "y" not in dataset.columns:
        raise ValueError("dataset must contain 'y' column")

    service = {"child_id", "domain", "cut", "y"}
    numeric_cols = [
        c for c in dataset.columns
        if c not in service and pd.api.types.is_numeric_dtype(dataset[c])
    ]

    domain_dummies = pd.get_dummies(dataset["domain"], prefix="domain")
    X = pd.concat([dataset[numeric_cols].reset_index(drop=True),
                   domain_dummies.reset_index(drop=True)], axis=1)

    y = dataset["y"].reset_index(drop=True)
    groups = dataset["child_id"].reset_index(drop=True)

    return X, y, groups