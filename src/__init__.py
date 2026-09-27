"""Пакет src для анализа прогресса детей."""
import pandas as pd


def open_excel(file_path):
    """Читает Excel-файл с сессиями детей."""
    return pd.read_excel(file_path)