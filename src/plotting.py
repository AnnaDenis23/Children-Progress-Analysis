"""Plotly-графики для Streamlit.

Каждая функция возвращает plotly.graph_objects.Figure — Streamlit умеет
рендерить их напрямую через st.plotly_chart(fig).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import STAGNATION_MIN_DAYS


# ---------------------------------------------------------------------------
# 1. Траектория одного ребёнка по одному навыку
# ---------------------------------------------------------------------------
def plot_trajectory(
    df: pd.DataFrame,
    child_id: str,
    domain: str,
    plateau_days: int | None = None,
) -> go.Figure | None:
    """Линия score по времени + маркеры улучшений + подсветка плато.

    Parameters
    ----------
    df : pd.DataFrame
        Полный (очищенный) df.
    child_id, domain : str
    plateau_days : int | None
        Если задано — подсвечиваем последний отрезок "дней без прогресса".

    Returns
    -------
    go.Figure | None
        None, если для пары нет данных.
    """
    part = df[(df["child_id"] == child_id) & (df["domain"] == domain)].copy()
    if part.empty:
        return None

    part = part.sort_values("session_date").reset_index(drop=True)
    part["score_diff"] = part["assessment_score"].diff()

    fig = go.Figure()

    # Основная линия
    fig.add_trace(go.Scatter(
        x=part["session_date"],
        y=part["assessment_score"],
        mode="lines+markers",
        name="Score",
        line=dict(color="#2E86AB", width=2),
        marker=dict(size=9),
        hovertemplate="%{x|%d.%m.%Y}<br>score = %{y}<extra></extra>",
    ))

    # Точки улучшения — зелёные
    improved = part[part["score_diff"] > 0]
    if not improved.empty:
        fig.add_trace(go.Scatter(
            x=improved["session_date"],
            y=improved["assessment_score"],
            mode="markers",
            name="Улучшение",
            marker=dict(color="#2ECC71", size=14, symbol="star",
                        line=dict(color="white", width=1)),
            hovertemplate="Улучшение<br>%{x|%d.%m.%Y}<br>score = %{y}<extra></extra>",
        ))

    # Подсветка текущего плато
    if plateau_days and plateau_days >= STAGNATION_MIN_DAYS:
        last_date = part["session_date"].iloc[-1]
        start_plateau = last_date - pd.Timedelta(days=int(plateau_days))
        fig.add_vrect(
            x0=start_plateau,
            x1=last_date,
            fillcolor="red",
            opacity=0.10,
            line_width=0,
            annotation_text=f"{int(plateau_days)} дн без прогресса",
            annotation_position="top left",
            annotation=dict(font_size=11, font_color="#B03A2E"),
        )

    fig.update_layout(
        title=f"{child_id} — {domain}",
        xaxis_title="Дата",
        yaxis_title="Score",
        hovermode="x unified",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Все навыки одного ребёнка (мульти-линия)
# ---------------------------------------------------------------------------
def plot_child_all_domains(df: pd.DataFrame, child_id: str) -> go.Figure | None:
    """Все домены одного ребёнка на одном графике."""
    part = df[df["child_id"] == child_id].copy()
    if part.empty:
        return None

    part = part.sort_values("session_date")

    fig = px.line(
        part,
        x="session_date",
        y="assessment_score",
        color="domain",
        markers=True,
        title=f"{child_id} — все навыки",
    )
    fig.update_layout(
        xaxis_title="Дата",
        yaxis_title="Score",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Bar-чарт дней без прогресса (все пациенты)
# ---------------------------------------------------------------------------
def plot_days_without_progress(report: pd.DataFrame, top_n: int = 20) -> go.Figure | None:
    """Горизонтальный bar: топ-N детей по дням без прогресса."""
    if report.empty:
        return None

    data = report.sort_values("days_without_progress", ascending=False).head(top_n).copy()
    data["label"] = data["child_id"] + " · " + data["domain"]

    color_map = {"low": "#2ECC71", "medium": "#F1C40F", "high": "#E74C3C"}

    fig = px.bar(
        data,
        x="days_without_progress",
        y="label",
        orientation="h",
        color="risk_level",
        color_discrete_map=color_map,
        title=f"Топ-{top_n}: дни без прогресса",
        labels={"days_without_progress": "Дней без прогресса", "label": ""},
    )
    fig.update_layout(
        template="plotly_white",
        height=max(300, 24 * len(data) + 100),
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis=dict(autorange="reversed"),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Scatter: sessions/month vs days_without_progress
# ---------------------------------------------------------------------------
def plot_intensity_vs_stagnation(report: pd.DataFrame) -> go.Figure | None:
    """Scatter: интенсивность занятий vs дни без прогресса."""
    if report.empty or "sessions_last_28_days" not in report.columns:
        return None

    color_map = {"low": "#2ECC71", "medium": "#F1C40F", "high": "#E74C3C"}

    fig = px.scatter(
        report,
        x="sessions_last_28_days",
        y="days_without_progress",
        color="risk_level",
        color_discrete_map=color_map,
        hover_data=["child_id", "domain", "last_score"],
        title="Интенсивность (28 дней) vs дни без прогресса",
        labels={
            "sessions_last_28_days": "Занятий за 28 дней",
            "days_without_progress": "Дней без прогресса",
        },
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Bar: важность признаков (SHAP)
# ---------------------------------------------------------------------------
def plot_shap_importance(importance_df: pd.DataFrame, title: str = "Важность признаков") -> go.Figure:
    """Горизонтальный bar по mean(|SHAP|)."""
    data = importance_df.sort_values("mean_abs_shap", ascending=True)

    fig = px.bar(
        data,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=title,
        labels={"mean_abs_shap": "mean(|SHAP|)", "feature": ""},
        color="mean_abs_shap",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template="plotly_white",
        height=max(280, 26 * len(data) + 100),
        margin=dict(l=40, r=20, t=60, b=40),
        coloraxis_showscale=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Waterfall-подобный bar для одного пациента (SHAP по признакам)
# ---------------------------------------------------------------------------
def plot_shap_patient(shap_df: pd.DataFrame, child_id: str, domain: str) -> go.Figure:
    """Bar: вклад признаков в риск для одного пациента."""
    data = shap_df.sort_values("shap_value")

    colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in data["shap_value"]]

    fig = go.Figure(go.Bar(
        x=data["shap_value"],
        y=data["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in data["shap_value"]],
        textposition="outside",
        hovertemplate="%{y}<br>SHAP = %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Вклад признаков в риск — {child_id} / {domain}",
        xaxis_title="SHAP (положительные → риск, отрицательные → защита)",
        yaxis_title="",
        template="plotly_white",
        height=max(280, 30 * len(data) + 100),
        margin=dict(l=40, r=60, t=60, b=40),
    )
    return fig