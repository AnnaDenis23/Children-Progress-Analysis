import pandas as pd


def clean_data(df):
    df = df.copy()

    df.columns = [col.strip() for col in df.columns]

    df["session_date"] = pd.to_datetime(df["session_date"], dayfirst=True, errors="coerce")
    df["assessment_score"] = pd.to_numeric(df["assessment_score"], errors="coerce")

    df["comment"] = df["comment"].fillna("")
    df["progress_flag"] = df["progress_flag"].fillna("")
    df["specialist_type"] = df["specialist_type"].fillna("")

    return df


def fix_progress_and_specialist(df):
    df = df.copy()

    specialist_words = ["логопед", "дефектолог", "па", "психолог", "тьютор"]

    new_progress = []
    new_specialist = []

    for _, row in df.iterrows():
        progress = str(row["progress_flag"]).strip().lower()
        specialist = str(row["specialist_type"]).strip().lower()

        if progress == "nan":
            progress = ""
        if specialist == "nan":
            specialist = ""

        if progress == "импровед":
            progress = "improved"
        if progress == "стагнант":
            progress = "stagnant"

        if progress in specialist_words:
            if specialist == "":
                specialist = progress
            progress = ""

        if specialist == "":
            specialist = "unknown"

        if progress == "":
            progress = "unknown"

        new_progress.append(progress)
        new_specialist.append(specialist)

    df["progress_flag"] = new_progress
    df["specialist_type"] = new_specialist

    return df


def check_columns(df):
    needed_columns = [
        "child_id",
        "age",
        "diagnosis",
        "domain",
        "session_date",
        "assessment_score",
        "comment",
        "progress_flag",
        "specialist_type",
    ]

    missing_columns = []

    for col in needed_columns:
        if col not in df.columns:
            missing_columns.append(col)

    return missing_columns


def check_child_id(df):
    bad_rows = df[~df["child_id"].astype(str).str.match(r"^СП\d+$", na=False)]
    return bad_rows


def check_score(df):
    bad_rows = df[
        (df["assessment_score"].isna()) |
        (df["assessment_score"] < 0) |
        (df["assessment_score"] > 10)
    ]
    return bad_rows


def check_dates(df):
    bad_rows = df[df["session_date"].isna()]
    return bad_rows