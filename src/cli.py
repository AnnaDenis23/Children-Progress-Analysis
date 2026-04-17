import os

from src import open_excel
from src.validation import (
    clean_data,
    fix_progress_and_specialist,
    check_columns,
    check_child_id,
    check_score,
    check_dates,
)
from src.analysis import add_auto_progress_flag, detect_stagnation
from src.reporting import save_csv, save_excel, make_summary, save_summary, make_plot


def main():
    input_file = "data/children_sessions.xlsx"
    output_folder = "outputs"
    plots_folder = os.path.join(output_folder, "plots")

    report_csv_path = os.path.join(output_folder, "stagnation_report.csv")
    report_excel_path = os.path.join(output_folder, "stagnation_report.xlsx")
    summary_path = os.path.join(output_folder, "summary.md")

    min_days = 28

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    print("1. Открываю Excel...")
    print("Файл:", input_file)
    df = open_excel(input_file)

    print("2. Чищу данные...")
    df = clean_data(df)

    print("3. Исправляю progress_flag и specialist_type...")
    df = fix_progress_and_specialist(df)

    print("4. Проверяю колонки...")
    missing_columns = check_columns(df)
    if missing_columns:
        print("Не хватает колонок:", missing_columns)
        return

    print("5. Проверяю child_id...")
    bad_child_id = check_child_id(df)
    print("Неправильных child_id:", len(bad_child_id))

    print("6. Проверяю score...")
    bad_score = check_score(df)
    print("Ошибок в score:", len(bad_score))

    print("7. Проверяю даты...")
    bad_dates = check_dates(df)
    print("Ошибок в датах:", len(bad_dates))

    if len(bad_child_id) > 0 or len(bad_score) > 0 or len(bad_dates) > 0:
        print("Есть ошибки в данных. Сначала исправь их.")
        return

    print("8. Считаю автоматический progress_flag...")
    df = add_auto_progress_flag(df)

    print("9. Ищу stagnation...")
    report = detect_stagnation(df, min_days=min_days)

    print("10. Сохраняю CSV отчет...")
    print("Сохраняю сюда:", report_csv_path)
    save_csv(report, report_csv_path)

    print("11. Сохраняю Excel отчет...")
    print("Сохраняю сюда:", report_excel_path)
    save_excel(report, report_excel_path)

    print("12. Делаю summary...")
    summary_text = make_summary(report)
    print("Сохраняю summary сюда:", summary_path)
    save_summary(summary_text, summary_path)

    print("13. Строю графики...")
    for _, row in report.iterrows():
        make_plot(
            df,
            row["child_id"],
            row["domain"],
            plots_folder
        )

    print("Готово!")
    print("Все результаты лежат в папке:", output_folder)
    print("Папка с графиками:", plots_folder)


if __name__ == "__main__":
    main()