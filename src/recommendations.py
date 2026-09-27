"""Правила генерации рекомендаций по каждому (child_id, domain).

Каждая рекомендация — dict с полями:
    priority  : critical | high | medium | low
    category  : intensity | methodology | specialist | monitoring | context | engagement
    text      : текст для врача (русский)
    evidence  : числовое обоснование
    based_on  : список признаков, на которых основана рекомендация
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from src.config import (
    FREQUENT_PLATEAUS,
    FREQUENT_SPECIALIST_CHANGES,
    HIGH_VOLATILITY_STD,
    LONG_PLATEAU_DAYS,
    MIN_SESSIONS_PER_MONTH,
    NEGATIVE_KEYWORDS,
)

Recommendation = dict


# ---------------------------------------------------------------------------
# Каждое правило — чистая функция row -> Recommendation | None
# ---------------------------------------------------------------------------
def _rule_low_intensity(row: pd.Series) -> Recommendation | None:
    if row["sessions_per_month"] < MIN_SESSIONS_PER_MONTH:
        return {
            "priority": "high",
            "category": "intensity",
            "text": (
                f"Мало занятий: {row['sessions_per_month']:.1f}/мес "
                f"(норма ≥ {MIN_SESSIONS_PER_MONTH}). "
                "Рекомендуется увеличить частоту."
            ),
            "evidence": f"sessions_per_month = {row['sessions_per_month']:.2f}",
            "based_on": ["sessions_per_month"],
        }
    return None


def _rule_long_plateau(row: pd.Series) -> Recommendation | None:
    if row["current_plateau_days"] >= LONG_PLATEAU_DAYS:
        return {
            "priority": "critical",
            "category": "methodology",
            "text": (
                f"Длительное плато: {int(row['current_plateau_days'])} дней без роста. "
                "Требуется пересмотр программы и методики."
            ),
            "evidence": f"current_plateau_days = {int(row['current_plateau_days'])}",
            "based_on": ["current_plateau_days"],
        }
    return None


def _rule_frequent_plateaus(row: pd.Series) -> Recommendation | None:
    if row["plateaus_count"] >= FREQUENT_PLATEAUS:
        return {
            "priority": "high",
            "category": "methodology",
            "text": (
                f"Ребёнок часто застревает: {int(row['plateaus_count'])} плато "
                "в истории. Нужна адаптивная методика и более дробные цели."
            ),
            "evidence": f"plateaus_count = {int(row['plateaus_count'])}",
            "based_on": ["plateaus_count"],
        }
    return None


def _rule_specialist_changes(row: pd.Series) -> Recommendation | None:
    if row["specialist_changes"] >= FREQUENT_SPECIALIST_CHANGES:
        return {
            "priority": "high",
            "category": "specialist",
            "text": (
                f"Частая смена специалистов: {int(row['specialist_changes'])} раз. "
                "Обсудить преемственность программы между специалистами."
            ),
            "evidence": f"specialist_changes = {int(row['specialist_changes'])}",
            "based_on": ["specialist_changes"],
        }
    return None


def _rule_high_volatility(row: pd.Series) -> Recommendation | None:
    if row["score_std"] >= HIGH_VOLATILITY_STD:
        return {
            "priority": "medium",
            "category": "monitoring",
            "text": (
                f"Высокая нестабильность score (std = {row['score_std']:.2f}). "
                "Проверить внешние факторы: сон, здоровье, эмоциональное состояние."
            ),
            "evidence": f"score_std = {row['score_std']:.2f}",
            "based_on": ["score_std"],
        }
    return None


def _rule_negative_comments(row: pd.Series) -> Recommendation | None:
    if row["negative_keywords_count"] >= 2:
        return {
            "priority": "medium",
            "category": "context",
            "text": (
                f"В комментариях {int(row['negative_keywords_count'])} негативных маркеров. "
                "Обратить внимание на мотивацию и вовлечённость."
            ),
            "evidence": f"negative_keywords_count = {int(row['negative_keywords_count'])}",
            "based_on": ["negative_keywords_count"],
        }
    return None


def _rule_positive_comments(row: pd.Series) -> Recommendation | None:
    if row["positive_keywords_count"] >= 2 and row["current_plateau_days"] >= 28:
        return {
            "priority": "low",
            "category": "engagement",
            "text": (
                "В комментариях позитивные маркеры несмотря на плато — "
                "возможно, ребёнок готов к следующему шагу."
            ),
            "evidence": f"positive_keywords_count = {int(row['positive_keywords_count'])}",
            "based_on": ["positive_keywords_count", "current_plateau_days"],
        }
    return None


def _rule_no_progress_at_all(row: pd.Series) -> Recommendation | None:
    if row["total_improvements"] == 0 and row["total_sessions"] >= 3:
        return {
            "priority": "critical",
            "category": "methodology",
            "text": (
                "Ни одного улучшения за всю историю занятий. "
                "Срочный пересмотр целей и диагностика барьеров."
            ),
            "evidence": f"total_improvements = 0, total_sessions = {int(row['total_sessions'])}",
            "based_on": ["total_improvements", "total_sessions"],
        }
    return None


RULES: list[Callable[[pd.Series], Recommendation | None]] = [
    _rule_no_progress_at_all,
    _rule_long_plateau,
    _rule_low_intensity,
    _rule_frequent_plateaus,
    _rule_specialist_changes,
    _rule_high_volatility,
    _rule_negative_comments,
    _rule_positive_comments,
]


def build_recommendations(features_df: pd.DataFrame) -> pd.DataFrame:
    """Строит рекомендации по каждой паре (child_id, domain).

    Returns
    -------
    pd.DataFrame
        Колонки: child_id, domain, recommendations (list[dict]),
        priority_max (str), recommendations_count (int).
    """
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame")

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    rows = []
    for _, row in features_df.iterrows():
        recs = [r for rule in RULES if (r := rule(row)) is not None]
        recs.sort(key=lambda r: priority_order.get(r["priority"], 99))

        rows.append({
            "child_id": row["child_id"],
            "domain": row["domain"],
            "recommendations": recs,
            "recommendations_count": len(recs),
            "priority_max": recs[0]["priority"] if recs else "none",
        })

    return pd.DataFrame(rows)


def recommendations_to_text(recs: list[Recommendation]) -> str:
    """Человекочитаемый текст блока рекомендаций (для summary.md / Streamlit)."""
    if not recs:
        return "Рекомендации не требуются — динамика в норме."

    icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    lines = []
    for r in recs:
        icon = icons.get(r["priority"], "⚪")
        lines.append(f"{icon} [{r['category']}] {r['text']}")
        lines.append(f"    └ основание: {r['evidence']}")
    return "\n".join(lines)