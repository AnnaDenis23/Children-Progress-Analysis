import numpy as np
import pandas as pd
import pytest

from src.features import compute_features
from src.ml.dataset import build_dataset, prepare_xy
from src.ml.train import cross_validate_model, train_all, _make_rf
from src.ml.explain import explain_patient, predict_patient, global_feature_importance


def _make_synthetic_df(n_children: int = 8, n_sessions: int = 6) -> pd.DataFrame:
    """Синтетика: половина детей — плато, половина — прогресс."""
    rng = np.random.default_rng(42)
    rows = []
    base_date = pd.Timestamp("2026-01-01")

    for i in range(n_children):
        child = f"СП{i:02d}"
        is_stagnant = i % 2 == 0
        score = 3
        for j in range(n_sessions):
            if not is_stagnant and j > 0:
                score += 1  # рост
            rows.append({
                "child_id": child,
                "domain": "Listening",
                "session_date": base_date + pd.Timedelta(days=15 * j),
                "assessment_score": float(score),
                "comment": "ok" if not is_stagnant else "сложно",
                "specialist_type": "логопед",
            })

    return pd.DataFrame(rows)


# --- dataset ---

def test_build_dataset_returns_expected_columns():
    df = _make_synthetic_df()
    ds = build_dataset(df, min_history=4, min_future_days=14)
    assert not ds.empty
    assert "y" in ds.columns
    assert "child_id" in ds.columns
    assert "cut" in ds.columns


def test_build_dataset_has_both_classes():
    df = _make_synthetic_df()
    ds = build_dataset(df, min_history=4, min_future_days=14)
    assert set(ds["y"].unique()) == {0, 1}


def test_prepare_xy_shapes_match():
    df = _make_synthetic_df()
    ds = build_dataset(df, min_history=4, min_future_days=14)
    X, y, groups = prepare_xy(ds)
    assert len(X) == len(y) == len(groups)
    assert "domain_Listening" in X.columns


def test_build_dataset_wrong_type():
    with pytest.raises(TypeError):
        build_dataset("not a df")


# --- train ---

def test_cross_validate_returns_metrics():
    df = _make_synthetic_df(n_children=8, n_sessions=6)
    ds = build_dataset(df, min_history=4, min_future_days=14)
    X, y, groups = prepare_xy(ds)

    result = cross_validate_model(_make_rf(), "RF", X, y, groups)
    assert 0.0 <= result.roc_auc_mean <= 1.0
    assert result.model_name == "RF"


def test_train_all_returns_two_models():
    df = _make_synthetic_df()
    ds = build_dataset(df, min_history=4, min_future_days=14)
    X, y, groups = prepare_xy(ds)

    metrics, fitted = train_all(X, y, groups)
    assert set(fitted.keys()) == {"RandomForest", "LogisticRegression"}
    assert len(metrics) == 2
    assert "roc_auc" in metrics.columns


def test_cv_requires_two_groups():
    df = _make_synthetic_df(n_children=1, n_sessions=8)
    ds = build_dataset(df, min_history=4, min_future_days=14)
    X, y, groups = prepare_xy(ds)
    with pytest.raises(ValueError):
        cross_validate_model(_make_rf(), "RF", X, y, groups)


# --- explain ---

def test_predict_and_explain_single_patient():
    df = _make_synthetic_df()
    ds = build_dataset(df, min_history=4, min_future_days=14)
    X, y, groups = prepare_xy(ds)

    model = _make_rf()
    model.fit(X, y)

    X_one = X.iloc[[0]]
    proba = predict_patient(model, X_one)
    assert 0.0 <= proba <= 1.0

    top = explain_patient(model, X_one, top_n=3)
    assert len(top) == 3
    assert "direction" in top.columns


def test_global_feature_importance():
    df = _make_synthetic_df()
    ds = build_dataset(df, min_history=4, min_future_days=14)
    X, y, groups = prepare_xy(ds)

    model = _make_rf()
    model.fit(X, y)

    imp = global_feature_importance(model, X, top_n=5)
    assert len(imp) == 5
    assert imp["mean_abs_shap"].is_monotonic_decreasing