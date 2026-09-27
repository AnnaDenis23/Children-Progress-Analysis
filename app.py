"""Streamlit-интерфейс для анализа прогресса детей.

Запуск:
    streamlit run app.py
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import open_excel
from src.analysis import add_auto_progress_flag, detect_stagnation
from src.config import (
    FREQUENT_PLATEAUS,
    FREQUENT_SPECIALIST_CHANGES,
    HIGH_VOLATILITY_STD,
    MIN_SESSIONS_PER_MONTH,
    STAGNATION_MIN_DAYS,
)
from src.features import compute_features
from src.ml.dataset import build_dataset, prepare_xy
from src.ml.explain import (
    explain_patient,
    global_feature_importance,
    predict_patient,
)
from src.ml.train import train_all
from src.plotting import (
    plot_child_all_domains,
    plot_days_without_progress,
    plot_intensity_vs_stagnation,
    plot_shap_importance,
    plot_shap_patient,
    plot_trajectory,
)
from src.recommendations import build_recommendations, recommendations_to_text
from src.validation import (
    check_child_id,
    check_dates,
    check_score,
    clean_data,
    fix_progress_and_specialist,
)

# ---------------------------------------------------------------------------
# Настройки страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Children Progress Analysis",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Анализ прогресса детей")
st.caption(
    "Инструмент для выявления застоя в развитии и формирования рекомендаций. "
    "⚠️ Не является медицинским заключением."
)

# ---------------------------------------------------------------------------
# Колонки исходных данных
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "child_id", "age", "diagnosis", "domain",
    "session_date", "assessment_score", "comment",
    "progress_flag", "specialist_type",
]

DOMAINS = ["Listening", "Social", "Verbal_Request", "Motor", "Play", "Academic"]
SPECIALISTS = ["логопед", "дефектолог", "психолог", "па", "тьютор", "unknown"]


# ---------------------------------------------------------------------------
# Кэш загрузки Excel
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_excel_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def _load_excel_path(path: str) -> pd.DataFrame:
    return open_excel(path)


# ---------------------------------------------------------------------------
# Подготовка (clean + fix + flags)
# ---------------------------------------------------------------------------
def _prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df_raw)
    df = fix_progress_and_specialist(df)
    df = add_auto_progress_flag(df)
    return df


# ---------------------------------------------------------------------------
# Сессия: сохраняем df между rerun'ами
# ---------------------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# ===========================================================================
# ВКЛАДКИ
# ===========================================================================
tab_data, tab_overview, tab_patient, tab_ml = st.tabs(
    ["📥 Данные", "📊 Обзор", "🧒 Пациент", "🤖 ML (PoC)"]
)

# ---------------------------------------------------------------------------
# 1. ДАННЫЕ
# ---------------------------------------------------------------------------
with tab_data:
    st.header("Источник данных")

    source = st.radio(
        "Как загрузить данные?",
        options=["Загрузить Excel", "Ввести вручную", "Демо-данные из data/"],
        horizontal=True,
    )

    if source == "Загрузить Excel":
        uploaded = st.file_uploader("Excel-файл (.xlsx)", type=["xlsx"])
        if uploaded is not None:
            try:
                df_raw = _load_excel_bytes(uploaded.getvalue())
                st.session_state.df_raw = df_raw
                st.success(f"Загружено строк: {len(df_raw)}")
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")

    elif source == "Демо-данные из data/":
        if st.button("Загрузить data/children_sessions.xlsx"):
            try:
                df_raw = _load_excel_path("data/children_sessions.xlsx")
                st.session_state.df_raw = df_raw
                st.success(f"Загружено строк: {len(df_raw)}")
            except Exception as e:
                st.error(f"Файл не найден или не читается: {e}")

    else:  # ручной ввод
        st.info(
            "Отредактируйте таблицу ниже. Кнопка «+» внизу добавляет строку. "
            "Значения `session_date` — в формате `YYYY-MM-DD`."
        )

        if st.session_state.df_raw is None:
            st.session_state.df_raw = pd.DataFrame(
                [{c: "" for c in REQUIRED_COLUMNS}]
            )

        edited = st.data_editor(
            st.session_state.df_raw,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_editor",
            column_config={
                "session_date": st.column_config.TextColumn("session_date"),
                "assessment_score": st.column_config.NumberColumn(
                    "assessment_score", min_value=0, max_value=10, step=1,
                ),
                "domain": st.column_config.SelectboxColumn(
                    "domain", options=DOMAINS,
                ),
                "specialist_type": st.column_config.SelectboxColumn(
                    "specialist_type", options=SPECIALISTS,
                ),
            },
        )
        st.session_state.df_raw = edited

    # --- Предпросмотр и валидация ---
    if st.session_state.df_raw is not None and len(st.session_state.df_raw) > 0:
        st.divider()
        st.subheader("Предпросмотр")

        df_raw = st.session_state.df_raw.copy()

        try:
            df_prepared = _prepare(df_raw)
        except Exception as e:
            st.error(f"Ошибка подготовки данных: {e}")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Строк", len(df_prepared))
        col2.metric("Детей", df_prepared["child_id"].nunique())
        col3.metric("Навыков", df_prepared["domain"].nunique())

        bad_id = check_child_id(df_prepared)
        bad_score = check_score(df_prepared)
        bad_dates = check_dates(df_prepared)
        col4.metric("Проблем", len(bad_id) + len(bad_score) + len(bad_dates))

        with st.expander("Показать очищенные данные"):
            st.dataframe(df_prepared, use_container_width=True)

        if len(bad_id) or len(bad_score) or len(bad_dates):
            st.warning(
                f"⚠️ Найдены ошибки: child_id={len(bad_id)}, "
                f"score={len(bad_score)}, dates={len(bad_dates)}. "
                "Строки с ошибками будут исключены из анализа."
            )

        # Убираем плохие строки
        df_clean = df_prepared.drop(
            index=set(bad_id.index) | set(bad_score.index) | set(bad_dates.index)
        ).copy()

        st.session_state.df_clean = df_clean
    else:
        st.session_state.df_clean = None

# ---------------------------------------------------------------------------
# Проверка: есть ли данные для анализа
# ---------------------------------------------------------------------------
df_clean = st.session_state.get("df_clean", None)

if df_clean is None or len(df_clean) == 0:
    with tab_overview:
        st.info("Сначала загрузите данные на вкладке «📥 Данные».")
    with tab_patient:
        st.info("Сначала загрузите данные на вкладке «📥 Данные».")
    with tab_ml:
        st.info("Сначала загрузите данные на вкладке «📥 Данные».")
    st.stop()

# ---------------------------------------------------------------------------
# Общие вычисления (один раз, для всех вкладок)
# ---------------------------------------------------------------------------
with st.spinner("Считаю признаки и stagnation..."):
    features_df = compute_features(df_clean)
    recs_df = build_recommendations(features_df)
    report = detect_stagnation(df_clean, min_days=STAGNATION_MIN_DAYS)

    # Объединяем report с рекомендациями
    if not report.empty:
        report = report.merge(
            recs_df[["child_id", "domain", "recommendations",
                     "recommendations_count", "priority_max"]],
            on=["child_id", "domain"],
            how="left",
        )
    else:
        report["recommendations"] = []
        report["recommendations_count"] = 0
        report["priority_max"] = "none"

# ---------------------------------------------------------------------------
# 2. ОБЗОР
# ---------------------------------------------------------------------------
with tab_overview:
    st.header("Обзор всех случаев")

    if report.empty:
        st.success("🎉 Случаев застоя не обнаружено.")
    else:
        # Метрики
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего случаев", len(report))
        c2.metric("🔴 High", int((report["risk_level"] == "high").sum()))
        c3.metric("🟡 Medium", int((report["risk_level"] == "medium").sum()))
        c4.metric("🟢 Low", int((report["risk_level"] == "low").sum()))

        st.divider()

        # Фильтры
        with st.expander("Фильтры", expanded=True):
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                risk_filter = st.multiselect(
                    "Risk level",
                    options=["high", "medium", "low"],
                    default=["high", "medium", "low"],
                )
            with fcol2:
                status_filter = st.multiselect(
                    "Stagnation status",
                    options=sorted(report["stagnation_status"].dropna().unique()),
                    default=sorted(report["stagnation_status"].dropna().unique()),
                )
            with fcol3:
                domain_filter = st.multiselect(
                    "Domain",
                    options=sorted(report["domain"].unique()),
                    default=sorted(report["domain"].unique()),
                )

        filtered = report[
            report["risk_level"].isin(risk_filter)
            & report["stagnation_status"].isin(status_filter)
            & report["domain"].isin(domain_filter)
        ].copy()

        st.write(f"Показано: **{len(filtered)}** случаев")

        # Таблица
        display_cols = [
            "child_id", "domain", "days_without_progress", "risk_level",
            "stagnation_status", "priority_max", "last_score",
            "sessions_last_28_days", "specialist_type",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].sort_values(
                "days_without_progress", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Графики
        st.divider()
        st.subheader("Визуализация")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_bar = plot_days_without_progress(filtered, top_n=20)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        with col_b:
            fig_scatter = plot_intensity_vs_stagnation(filtered)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Экспорт
        st.divider()
        csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Скачать отчёт (CSV)",
            data=csv_bytes,
            file_name="stagnation_report.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# 3. ПАЦИЕНТ
# ---------------------------------------------------------------------------
with tab_patient:
    st.header("Профиль пациента")

    children = sorted(df_clean["child_id"].unique())
    selected_child = st.selectbox("Выберите ребёнка", children)

    child_df = df_clean[df_clean["child_id"] == selected_child].copy()
    child_domains = sorted(child_df["domain"].unique())

    # Карточка пациента
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Возраст", int(child_df["age"].iloc[0]) if "age" in child_df else "—")
    col_b.metric("Диагноз", child_df["diagnosis"].iloc[0] if "diagnosis" in child_df else "—")
    col_c.metric("Навыков", len(child_domains))

    # График по всем навыкам
    fig_all = plot_child_all_domains(df_clean, selected_child)
    if fig_all is not None:
        st.plotly_chart(fig_all, use_container_width=True)

    st.divider()
    st.subheader("Разбор по навыкам")

    # По каждому навыку — свой блок
    child_report = report[report["child_id"] == selected_child] if not report.empty else pd.DataFrame()
    child_recs = recs_df[recs_df["child_id"] == selected_child]

    for dom in child_domains:
        with st.container(border=True):
            row_report = child_report[child_report["domain"] == dom]
            row_recs = child_recs[child_recs["domain"] == dom]

            c1, c2 = st.columns([2, 1])

            with c1:
                plateau = None
                if not row_report.empty:
                    plateau = int(row_report.iloc[0]["days_without_progress"])
                elif not features_df.empty:
                    feat_row = features_df[
                        (features_df["child_id"] == selected_child)
                        & (features_df["domain"] == dom)
                    ]
                    if not feat_row.empty:
                        plateau = int(feat_row.iloc[0]["current_plateau_days"])

                fig = plot_trajectory(df_clean, selected_child, dom, plateau_days=plateau)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if not row_report.empty:
                    r = row_report.iloc[0]
                    st.metric("Дней без прогресса", int(r["days_without_progress"]))
                    st.metric("Risk level", r["risk_level"].upper())
                    st.metric("Статус", r["stagnation_status"])
                else:
                    st.success("Прогресс идёт")
                    if not row_recs.empty:
                        feat = features_df[
                            (features_df["child_id"] == selected_child)
                            & (features_df["domain"] == dom)
                        ]
                        if not feat.empty:
                            st.metric("Дней без прогресса",
                                      int(feat.iloc[0]["current_plateau_days"]))

            # Рекомендации
            st.markdown("**Рекомендации:**")
            if not row_recs.empty and row_recs.iloc[0]["recommendations"]:
                for rec in row_recs.iloc[0]["recommendations"]:
                    icons = {"critical": "🔴", "high": "🟠",
                             "medium": "🟡", "low": "🟢"}
                    icon = icons.get(rec["priority"], "⚪")
                    st.markdown(
                        f"{icon} **[{rec['category']}]** {rec['text']}  \n"
                        f"<small style='color:#666'>Основание: {rec['evidence']}</small>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Рекомендации не требуются — динамика в норме.")

# ---------------------------------------------------------------------------
# 4. ML (PoC)
# ---------------------------------------------------------------------------
with tab_ml:
    st.header("ML-прогноз (Proof of Concept)")
    st.warning(
        "⚠️ Датасет мал. Результаты не предназначены для клинического "
        "применения без валидации на большем объёме данных."
    )

    if len(df_clean["child_id"].unique()) < 4:
        st.info("Для ML-анализа нужно минимум 4 ребёнка. Сейчас — "
                f"{len(df_clean['child_id'].unique())}.")
        st.stop()

    if st.button("🚀 Обучить модели и показать результаты"):
        with st.spinner("Обучаю модели..."):
            try:
                ds = build_dataset(df_clean, min_history=3, min_future_days=21)
                X, y, groups = prepare_xy(ds)
                metrics, fitted = train_all(X, y, groups)

                st.session_state.ml_fitted = fitted
                st.session_state.ml_X = X
                st.session_state.ml_metrics = metrics
                st.session_state.ml_ds = ds
            except Exception as e:
                st.error(f"Ошибка ML: {e}")
                st.stop()

    if "ml_fitted" in st.session_state:
        metrics = st.session_state.ml_metrics
        fitted = st.session_state.ml_fitted
        X = st.session_state.ml_X

        st.subheader("Метрики кросс-валидации (GroupKFold по child_id)")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        st.caption(
            f"Обучающих примеров: {len(X)} | "
            f"признаков: {X.shape[1]} | "
            f"уникальных детей: {groups_n}"
            if (groups_n := st.session_state.ml_ds['child_id'].nunique()) else ""
        )

        st.divider()
        st.subheader("Глобальная важность признаков (SHAP)")
        rf = fitted["RandomForest"]
        imp = global_feature_importance(rf, X, top_n=10)
        st.plotly_chart(plot_shap_importance(imp), use_container_width=True)

        st.divider()
        st.subheader("Прогноз для конкретного пациента")
        ml_child = st.selectbox(
            "Ребёнок", sorted(df_clean["child_id"].unique()), key="ml_child"
        )
        ml_domain = st.selectbox(
            "Навык",
            sorted(df_clean[df_clean["child_id"] == ml_child]["domain"].unique()),
            key="ml_domain",
        )

        feats_for_child = features_df[
            (features_df["child_id"] == ml_child)
            & (features_df["domain"] == ml_domain)
        ]
        if feats_for_child.empty:
            st.warning("Нет данных для этой пары.")
        else:
            # Собираем X_row той же структуры, что X
            try:
                row = feats_for_child.iloc[0].to_dict()
                row.pop("child_id", None)
                row.pop("domain", None)

                # one-hot для domain
                domain_dummies = pd.get_dummies(
                    pd.Series([ml_domain]), prefix="domain"
                )
                X_row = pd.concat(
                    [pd.DataFrame([row]), domain_dummies], axis=1
                )
                X_row = X_row.reindex(columns=X.columns, fill_value=0)

                proba = predict_patient(rf, X_row)
                st.metric(
                    "Вероятность застоя на след. месяц",
                    f"{proba * 100:.1f}%",
                )

                shap_df = explain_patient(rf, X_row, top_n=5)
                st.plotly_chart(
                    plot_shap_patient(shap_df, ml_child, ml_domain),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Не удалось посчитать прогноз: {e}")