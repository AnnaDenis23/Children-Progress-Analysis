import pandas as pd


def get_risk_level(days_without_progress):
    if days_without_progress >= 56:
        return "high"
    elif days_without_progress >= 42:
        return "medium"
    else:
        return "low"


def add_auto_progress_flag(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df = df.copy()
    df = df.sort_values(["child_id", "domain", "session_date"]).copy()

    all_groups = []

    for (child_id, domain), group in df.groupby(["child_id", "domain"], sort=False):
        group = group.sort_values("session_date").copy()
        group["score_diff"] = group["assessment_score"].diff()

        auto_flags = []

        for _, row in group.iterrows():
            if pd.isna(row["score_diff"]):
                auto_flags.append("unknown")
            elif row["score_diff"] > 0:
                auto_flags.append("improved")
            else:
                auto_flags.append("stagnant")

        group["auto_progress_flag"] = auto_flags
        all_groups.append(group)

    result = pd.concat(all_groups, ignore_index=True)
    return result


def get_last_28_days_info(group, last_session_date):
    start_date = last_session_date - pd.Timedelta(days=28)

    last_28 = group[group["session_date"] >= start_date].copy()
    last_28 = last_28.sort_values("session_date")

    sessions_count = len(last_28)
    scores_list = list(last_28["assessment_score"])

    progress_in_last_28_days = False
    if len(last_28) >= 2:
        diffs = last_28["assessment_score"].diff()
        if (diffs > 0).any():
            progress_in_last_28_days = True

    return sessions_count, scores_list, progress_in_last_28_days


def get_stagnation_status(days_without_progress, progress_in_last_28_days):
    if days_without_progress >= 28 and not progress_in_last_28_days:
        return "stagnant"

    if days_without_progress >= 28 and progress_in_last_28_days:
        return "plateau_risk"

    if days_without_progress < 28 and progress_in_last_28_days:
        return "recent_progress"

    return "stable"


def detect_stagnation(df, min_days=28):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(min_days, int):
        raise TypeError("min_days must be an integer")

    df = df.copy()
    df = df.sort_values(["child_id", "domain", "session_date"]).copy()

    results = []

    groups = df.groupby(["child_id", "domain"])

    for (child_id, domain), group in groups:
        group = group.sort_values("session_date").copy()

        if len(group) < 2:
            continue

        group["score_diff"] = group["assessment_score"].diff()

        improved_rows = group[group["score_diff"] > 0]

        if len(improved_rows) == 0:
            last_improvement_date = group["session_date"].min()
        else:
            last_improvement_date = improved_rows["session_date"].max()

        last_row = group.iloc[-1]
        last_session_date = last_row["session_date"]

        days_without_progress = (last_session_date - last_improvement_date).days

        sessions_last_28_days, scores_last_28_days, progress_in_last_28_days = get_last_28_days_info(
            group,
            last_session_date
        )

        stagnation_status = get_stagnation_status(
            days_without_progress,
            progress_in_last_28_days
        )

        if days_without_progress >= min_days or stagnation_status == "plateau_risk":
            results.append({
                "child_id": child_id,
                "domain": domain,
                "last_improvement_date": last_improvement_date,
                "last_session_date": last_session_date,
                "days_without_progress": days_without_progress,
                "last_score": last_row["assessment_score"],
                "recent_comment": last_row["comment"],
                "progress_flag_original": last_row["progress_flag"],
                "progress_flag_auto": last_row["auto_progress_flag"],
                "specialist_type": last_row["specialist_type"],
                "risk_level": get_risk_level(days_without_progress),
                "sessions_last_28_days": sessions_last_28_days,
                "scores_last_28_days": str(scores_last_28_days),
                "progress_in_last_28_days": progress_in_last_28_days,
                "stagnation_status": stagnation_status,
            })

    report = pd.DataFrame(results)
    return report