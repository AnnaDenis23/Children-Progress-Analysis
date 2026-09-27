import pandas as pd

from src.recommendations import build_recommendations, recommendations_to_text


def _features_row(**overrides):
    base = {
        "child_id": "СП01",
        "domain": "Listening",
        "total_sessions": 5,
        "total_improvements": 1,
        "improvement_rate": 0.25,
        "avg_days_between_improvements": 20.0,
        "score_trend": 0.01,
        "current_plateau_days": 10,
        "longest_plateau_days": 20,
        "plateaus_count": 0,
        "recovered_from_plateau": False,
        "sessions_per_month": 5.0,
        "days_since_last_session": 5.0,
        "gaps_over_14_days": 0,
        "score_std": 0.5,
        "score_range": 1.0,
        "specialists_count": 1,
        "specialist_changes": 0,
        "negative_keywords_count": 0,
        "positive_keywords_count": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_no_recommendations_for_healthy_case():
    df = _features_row()
    recs = build_recommendations(df)
    assert recs.iloc[0]["recommendations_count"] == 0
    assert recs.iloc[0]["priority_max"] == "none"


def test_long_plateau_triggers_critical():
    df = _features_row(current_plateau_days=70)
    recs = build_recommendations(df).iloc[0]["recommendations"]
    assert any(r["priority"] == "critical" for r in recs)


def test_low_intensity_triggers_high():
    df = _features_row(sessions_per_month=1.5)
    recs = build_recommendations(df).iloc[0]["recommendations"]
    assert any(r["category"] == "intensity" for r in recs)


def test_no_improvements_at_all_triggers_critical():
    df = _features_row(total_improvements=0, total_sessions=5)
    recs = build_recommendations(df).iloc[0]["recommendations"]
    assert any(r["category"] == "methodology" and r["priority"] == "critical" for r in recs)


def test_recommendations_sorted_by_priority():
    df = _features_row(
        current_plateau_days=70,
        sessions_per_month=1.0,
        specialist_changes=3,
    )
    recs = build_recommendations(df).iloc[0]["recommendations"]
    priorities = [r["priority"] for r in recs]
    assert priorities[0] in ("critical", "high")
    # critical должен идти раньше medium
    if "critical" in priorities and "medium" in priorities:
        assert priorities.index("critical") < priorities.index("medium")


def test_text_renderer():
    df = _features_row(current_plateau_days=70)
    recs = build_recommendations(df).iloc[0]["recommendations"]
    text = recommendations_to_text(recs)
    assert "🔴" in text or "🟠" in text
    assert "основание" in text