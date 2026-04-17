import pandas as pd
import pytest
from src.validation import(
    fix_progress_and_specialist,
    check_child_id,
    check_score,
    check_dates
)

from src.analysis import (
    add_auto_progress_flag,
    detect_stagnation,
    get_risk_level
)

class TestProject:
    def setup_method(self):
        self.df = pd.DataFrame({
            "child_id": ["СП01", "СП01", "СП01", "СП02", "BAD1"],
            "age": [4, 4, 4, 5, 6],
            "diagnosis": ["РАС", "РАС", "РАС", "РАС", "ЗПР"],
            "domain": ["Listening", "Listening", "Listening", "Social", "Motor"],
            "session_date": ["01.01.2026", "01.02.2026", "10.04.2026", "05.01.2026", "wrong_date"],
            "assessment_score": [5, 5, 5, 3, 12],
            "comment": ["a", "b", "c", "d", "e"],
            "progress_flag": ["импровед", "дефектолог", "", "stagnant", "unknown"],
            "specialist_type": ["", "", "логопед", "па", ""],
        })

        self.df["session_date"] = pd.to_datetime(
            self.df["session_date"],
            dayfirst=True,
            errors="coerce"
        )

    def test_fix_progress_and_specialist(self):
        result = fix_progress_and_specialist(self.df)

        assert type(result).__name__ == "DataFrame"

        assert result.loc[0, "progress_flag"] == "improved"
        assert result.loc[1, "specialist_type"] == "дефектолог"
        assert result.loc[1, "progress_flag"] == "unknown"
        assert result.loc[2, "specialist_type"] == "логопед"

    def test_check_child_id(self):
        bad_rows = check_child_id(self.df)

        assert type(bad_rows).__name__ == "DataFrame"
        assert len(bad_rows) == 1
        assert bad_rows.iloc[0]["child_id"] == "BAD1"

    def test_check_score(self):
        bad_rows = check_score(self.df)

        assert type(bad_rows).__name__ == "DataFrame"
        assert len(bad_rows) == 1
        assert bad_rows.iloc[0]["assessment_score"] == 12

    def test_check_dates(self):
        bad_rows = check_dates(self.df)

        assert type(bad_rows).__name__ == "DataFrame"
        assert len(bad_rows) == 1
        assert bad_rows.iloc[0]["child_id"] == "BAD1"

    def test_add_auto_progress_flag(self):
        clean_df = self.df.copy()
        clean_df = fix_progress_and_specialist(clean_df)

        # оставим только валидные строки для этого теста
        clean_df = clean_df[clean_df["child_id"] != "BAD1"].copy()

        result = add_auto_progress_flag(clean_df)

        assert type(result).__name__ == "DataFrame"
        assert "auto_progress_flag" in result.columns

        listening_rows = result[result["child_id"] == "СП01"]

        flags = list(listening_rows["auto_progress_flag"])
        assert flags[0] == "unknown"
        assert flags[1] == "stagnant"
        assert flags[2] == "stagnant"

    def test_detect_stagnation(self):
        df = pd.DataFrame({
            "child_id": ["СП01", "СП01", "СП01", "СП01"],
            "age": [4, 4, 4, 4],
            "diagnosis": ["РАС", "РАС", "РАС", "РАС"],
            "domain": ["Listening", "Listening", "Listening", "Listening"],
            "session_date": ["01.01.2026", "01.02.2026", "01.03.2026", "10.04.2026"],
            "assessment_score": [5, 5, 5, 5],
            "comment": ["a", "b", "c", "d"],
            "progress_flag": ["", "", "", ""],
            "specialist_type": ["логопед", "логопед", "логопед", "логопед"],
        })

        df["session_date"] = pd.to_datetime(df["session_date"], dayfirst=True)
        df = fix_progress_and_specialist(df)
        df = add_auto_progress_flag(df)

        report = detect_stagnation(df, min_days=28)

        assert type(report).__name__ == "DataFrame"
        assert len(report) == 1
        assert report.iloc[0]["child_id"] == "СП01"
        assert report.iloc[0]["domain"] == "Listening"
        assert report.iloc[0]["progress_flag_auto"] == "stagnant"

    def test_detect_stagnation_with_progress(self):
        df = pd.DataFrame({
            "child_id": ["СП01", "СП01", "СП01"],
            "age": [4, 4, 4],
            "diagnosis": ["РАС", "РАС", "РАС"],
            "domain": ["Verbal_Request", "Verbal_Request", "Verbal_Request"],
            "session_date": ["01.01.2026", "01.02.2026", "01.03.2026"],
            "assessment_score": [3, 4, 5],
            "comment": ["a", "b", "c"],
            "progress_flag": ["", "", ""],
            "specialist_type": ["логопед", "логопед", "логопед"],
        })

        df["session_date"] = pd.to_datetime(df["session_date"], dayfirst=True)
        df = fix_progress_and_specialist(df)
        df = add_auto_progress_flag(df)

        report = detect_stagnation(df, min_days=28)

        assert type(report).__name__ == "DataFrame"
        assert len(report) == 0

    def test_get_risk_level(self):
        assert get_risk_level(30) == "low"
        assert get_risk_level(45) == "medium"
        assert get_risk_level(60) == "high"

    def test_wrong_types(self):
        with pytest.raises(TypeError):
            detect_stagnation("not a dataframe", min_days=28)

        with pytest.raises(TypeError):
            add_auto_progress_flag("not a dataframe")