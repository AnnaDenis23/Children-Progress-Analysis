import pandas as pd
import pytest

from src.features import compute_features


def _make_df():
    return pd.DataFrame({
        "child_id": ["СП01"] * 4 + ["СП02"] * 3,
        "domain": ["Listening"] * 4 + ["Social"] * 3,
        "session_date": pd.to_datetime([
            "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-10",
            "2026-01-05", "2026-02-01", "2026-03-01",
        ]),
        "assessment_score": [5, 5, 5, 5, 3, 3, 4],
        "comment": ["сложно", "ok", "отказ", "ok",
                    "прогресс", "хорошо", "молодец"],
        "specialist_type": ["логопед"] * 4 + ["дефектолог", "психолог", "психолог"],
    })


def test_compute_features_returns_one_row_per_pair():
    df = _make_df()
    feats = compute_features(df)
    assert len(feats) == 2
    assert set(feats["child_id"]) == {"СП01", "СП02"}


def test_no_progress_child_has_plateau_and_zero_improvements():
    df = _make_df()
    feats = compute_features(df).set_index("child_id")
    сп01 = feats.loc["СП01"]
    assert сп01["total_improvements"] == 0
    assert сп01["current_plateau_days"] > 28
    assert сп01["score_trend"] == pytest.approx(0.0, abs=1e-9)


def test_progress_child_has_improvements_and_positive_trend():
    df = _make_df()
    feats = compute_features(df).set_index("child_id")
    сп02 = feats.loc["СП02"]
    assert сп02["total_improvements"] >= 1
    assert сп02["score_trend"] > 0


def test_context_features_count_keywords_and_specialists():
    df = _make_df()
    feats = compute_features(df).set_index("child_id")
    # СП02: 3 разных специалиста => 2 смены
    assert feats.loc["СП02", "specialist_changes"] >= 1
    assert feats.loc["СП02", "positive_keywords_count"] >= 2
    assert feats.loc["СП01", "negative_keywords_count"] >= 2


def test_wrong_types():
    with pytest.raises(TypeError):
        compute_features("not a df")

    with pytest.raises(ValueError):
        compute_features(pd.DataFrame({"child_id": ["x"]}))