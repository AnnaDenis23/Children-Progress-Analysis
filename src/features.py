"""Расчёт признаков (feature engineering) по парам child_id + domain.

Возвращает "профиль" по каждой паре: ~17 признаков из 5 слоёв:
1. Динамика траектории
2. Плато
3. Интенсивность занятий
4. Волатильность
5. Контекст (специалисты, комментарии)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    LONG_GAP_DAYS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Слой 1. Динамика траектории
# ---------------------------------------------------------------------------
def _trajectory_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("session_date")
    scores = group["assessment_score"].to_numpy(dtype=float)
    dates = group["session_date"].to_numpy()

    total_sessions = len(group)
    diffs = np.diff(scores)
    improvements_idx = np.where(diffs > 0)[0] + 1  # индексы "улучшений"
    total_improvements = int(len(improvements_idx))
    improvement_rate = total_improvements / max(total_sessions - 1, 1)

    if total_improvements >= 2:
        imp_dates = pd.to_datetime(dates[improvements_idx])
        gaps = np.diff(imp_dates).astype("timedelta64[D]").astype(float)
        avg_days_between_improvements = float(np.mean(gaps)) if len(gaps) else np.nan
    else:
        avg_days_between_improvements = np.nan

    # Наклон линейной регрессии score ~ день_от_старта.
    # Если score не меняется (ptp == 0) — наклон ровно 0.0,
    # чтобы избежать машинного шума вида -1.8e-17 от np.polyfit.
    if total_sessions >= 2:
        t0 = pd.to_datetime(dates[0])
        days = (pd.to_datetime(dates) - t0).days.to_numpy(dtype=float)
        if days.max() > 0 and np.ptp(scores) > 0:
            slope = float(np.polyfit(days, scores, 1)[0])
        else:
            slope = 0.0
    else:
        slope = 0.0

    return {
        "total_sessions": total_sessions,
        "total_improvements": total_improvements,
        "improvement_rate": round(improvement_rate, 4),
        "avg_days_between_improvements": avg_days_between_improvements,
        "score_trend": slope,
    }


# ---------------------------------------------------------------------------
# Слой 2. Плато
# ---------------------------------------------------------------------------
def _plateau_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("session_date")
    dates = group["session_date"].to_numpy()
    scores = group["assessment_score"].to_numpy(dtype=float)

    if len(group) == 0:
        return {
            "current_plateau_days": 0,
            "longest_plateau_days": 0,
            "plateaus_count": 0,
            "recovered_from_plateau": False,
        }

    # Индексы улучшений (первый элемент не улучшение)
    imp_idx = [i for i in range(1, len(scores)) if scores[i] > scores[i - 1]]
    last_date = pd.to_datetime(dates[-1])

    if imp_idx:
        last_improve_date = pd.to_datetime(dates[imp_idx[-1]])
    else:
        last_improve_date = pd.to_datetime(dates[0])

    current_plateau_days = (last_date - last_improve_date).days

    # Плато = интервал между улучшениями
    boundaries = [pd.to_datetime(dates[0])] + [pd.to_datetime(dates[i]) for i in imp_idx]
    plateaus = []
    for i in range(1, len(boundaries)):
        plateaus.append((boundaries[i] - boundaries[i - 1]).days)
    # добавляем текущее незакрытое плато
    plateaus.append((last_date - boundaries[-1]).days)

    longest_plateau_days = int(max(plateaus)) if plateaus else 0
    plateaus_count = sum(1 for p in plateaus if p >= 28)
    recovered_from_plateau = len(imp_idx) >= 2

    return {
        "current_plateau_days": int(current_plateau_days),
        "longest_plateau_days": longest_plateau_days,
        "plateaus_count": plateaus_count,
        "recovered_from_plateau": recovered_from_plateau,
    }


# ---------------------------------------------------------------------------
# Слой 3. Интенсивность занятий
# ---------------------------------------------------------------------------
def _intensity_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("session_date")
    dates = pd.to_datetime(group["session_date"]).to_numpy()
    last_date = pd.to_datetime(dates[-1])
    first_date = pd.to_datetime(dates[0])

    total_days = max((last_date - first_date).days, 1)
    sessions_per_month = len(group) / (total_days / 30.0)

    days_since_last_session = (last_date - pd.to_datetime(dates[-2])).days \
        if len(dates) >= 2 else np.nan

    gaps = np.diff(dates).astype("timedelta64[D]").astype(float) if len(dates) >= 2 else np.array([])
    gaps_over_14_days = int((gaps > LONG_GAP_DAYS).sum()) if len(gaps) else 0

    return {
        "sessions_per_month": round(float(sessions_per_month), 3),
        "days_since_last_session": days_since_last_session,
        "gaps_over_14_days": gaps_over_14_days,
    }


# ---------------------------------------------------------------------------
# Слой 4. Волатильность
# ---------------------------------------------------------------------------
def _volatility_features(group: pd.DataFrame) -> dict:
    scores = group["assessment_score"].astype(float)
    std = float(scores.std(ddof=0)) if len(scores) >= 2 else 0.0
    rng = float(scores.max() - scores.min()) if len(scores) else 0.0
    return {
        "score_std": round(std, 4),
        "score_range": round(rng, 4),
    }


# ---------------------------------------------------------------------------
# Слой 5. Контекст
# ---------------------------------------------------------------------------
def _count_keywords(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str):
        return 0
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)


def _context_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("session_date")

    specialists = group["specialist_type"].fillna("unknown").astype(str).tolist()
    unique_specialists = [s for s in set(specialists) if s and s != "unknown"]
    specialist_changes = sum(
        1 for i in range(1, len(specialists)) if specialists[i] != specialists[i - 1]
    )

    comments = group["comment"].fillna("").astype(str)
    neg = int(comments.map(lambda t: _count_keywords(t, NEGATIVE_KEYWORDS)).sum())
    pos = int(comments.map(lambda t: _count_keywords(t, POSITIVE_KEYWORDS)).sum())

    return {
        "specialists_count": len(unique_specialists),
        "specialist_changes": specialist_changes,
        "negative_keywords_count": neg,
        "positive_keywords_count": pos,
    }


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Считает признаки по каждой паре (child_id, domain).

    Parameters
    ----------
    df : pd.DataFrame
        Очищенный df с колонками child_id, domain, session_date,
        assessment_score, specialist_type, comment.

    Returns
    -------
    pd.DataFrame
        Одна строка на пару (child_id, domain) со всеми признаками.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = {"child_id", "domain", "session_date", "assessment_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    df = df.copy()
    df = df.dropna(subset=["session_date", "assessment_score"])
    df = df.sort_values(["child_id", "domain", "session_date"])

    rows = []
    for (child_id, domain), group in df.groupby(["child_id", "domain"], sort=False):
        if len(group) == 0:
            continue
        row = {"child_id": child_id, "domain": domain}
        row.update(_trajectory_features(group))
        row.update(_plateau_features(group))
        row.update(_intensity_features(group))
        row.update(_volatility_features(group))
        row.update(_context_features(group))
        rows.append(row)

    return pd.DataFrame(rows)