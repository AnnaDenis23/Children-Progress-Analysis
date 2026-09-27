# Children Progress Analysis

## Описание проекта

Проект посвящён анализу данных по занятиям детей с целью выявления случаев
отсутствия прогресса (stagnation), зон риска в развитии и формирования
персонализированных рекомендаций для специалистов.

Основные задачи:

* автоматически определять детей с длительным отсутствием прогресса
* выявлять риск застоя и слабую динамику после плато
* формировать понятные рекомендации по каждому ребёнку и навыку
* предоставлять интерактивный интерфейс для врача (Streamlit)
* демонстрировать применимость ML для прогноза риска (Proof of Concept)

Проект имитирует реальную задачу аналитика: работа с «грязными» данными,
feature engineering,rule-based рекомендации, ML-модель и UI для конечного
пользователя.

---

## Данные

Входной файл:
data/children_sessions.xlsx


Содержит информацию о занятиях:

| Колонка | Описание |
|---|---|
| `child_id` | идентификатор ребёнка (формат `СП01`) |
| `age` | возраст |
| `diagnosis` | диагноз |
| `domain` | навык (Listening, Social, Verbal_Request, …) |
| `session_date` | дата занятия |
| `assessment_score` | оценка прогресса (0–10) |
| `comment` | комментарий специалиста |
| `progress_flag` | исходный флаг прогресса (часто некорректный) |
| `specialist_type` | тип специалиста |

---

## Проблемы в данных и их решение

В исходных данных обнаружены ошибки:

* `progress_flag` иногда содержит тип специалиста
* `specialist_type` часто пустой
* значения на русском (`импровед`, `стагнант`)
* пропуски и некорректные значения в `score`, `session_date`, `child_id`

Реализована автоматическая очистка:

* исправление опечаток (`импровед` → `improved`, `стагнант` → `stagnant`)
* перенос значений из `progress_flag` в `specialist_type`, если это тип специалиста
* заполнение пропусков (`unknown`)
* валидация `child_id` (regex `^СП\d+$`), `score` (0–10), `session_date`

---

## Используемые технологии

* **Python 3.10+**
* **pandas, numpy** — обработка и анализ данных
* **scikit-learn** — ML (RandomForest, LogisticRegression, GroupKFold)
* **shap** — объяснение ML-прогнозов
* **Streamlit** — интерактивный UI для врача
* **Plotly** — интерактивные графики
* **matplotlib** — статические графики (CLI-пайплайн)
* **openpyxl** — работа с Excel
* **pytest** — тестирование
* **CLI (`python -m src.cli`)** — запуск пайплайна

---


Такое разделение упрощает поддержку, тестирование и расширение проекта.

---

## Логика анализа

### 1. Автоматический progress_flag

Исходный `progress_flag` ненадёжен, поэтому рассчитывается заново:

* score вырос → `improved`
* score не изменился или снизился → `stagnant`
* первая запись → `unknown`

### 2. Основное правило stagnation

Для каждой пары `child_id + domain`:

* определяется дата последнего улучшения
* рассчитывается количество дней без прогресса

Если `days_without_progress >= 28` → случай считается stagnation.

### 3. Анализ последних 28 дней

Дополнительно анализируются последние 28 дней:

* количество занятий
* значения score
* наличие прогресса

Поля: `sessions_last_28_days`, `scores_last_28_days`,
`progress_in_last_28_days`.

### 4. Выявление плато (plateau)

Если длительное время не было прогресса, и только в конце появился слабый
рост → случай помечается как `plateau_risk`.

---

## Feature Engineering (17 признаков)

Признаки считаются по каждой паре `(child_id, domain)` и разбиты на 5 слоёв:

### Слой 1. Динамика траектории

| Признак | Описание |
|---|---|
| `total_sessions` | всего занятий |
| `total_improvements` | сколько раз был рост score |
| `improvement_rate` | improvements / sessions |
| `avg_days_between_improvements` | средний интервал между улучшениями |
| `score_trend` | наклон линейной регрессии score по времени |

### Слой 2. Плато

| Признак | Описание |
|---|---|
| `current_plateau_days` | текущее плато |
| `longest_plateau_days` | самое длинное плато в истории |
| `plateaus_count` | количество плато ≥ 28 дней |
| `recovered_from_plateau` | были ли раньше выходы из плато |

### Слой 3. Интенсивность занятий

| Признак | Описание |
|---|---|
| `sessions_per_month` | средняя частота занятий |
| `days_since_last_session` | дней с последнего занятия |
| `gaps_over_14_days` | количество перерывов > 14 дней |

### Слой 4. Волатильность

| Признак | Описание |
|---|---|
| `score_std` | стандартное отклонение score |
| `score_range` | max − min |

### Слой 5. Контекст

| Признак | Описание |
|---|---|
| `specialists_count` | число уникальных специалистов |
| `specialist_changes` | сколько раз менялся специалист |
| `negative_keywords_count` | негативных маркеров в комментариях |
| `positive_keywords_count` | позитивных маркеров |

---

## Рекомендации (rule-based)

Каждая рекомендация — структурированный объект:

```python
{
    "priority": "critical | high | medium | low",
    "category": "intensity | methodology | specialist | monitoring | context | engagement",
    "text": "текст для врача (русский)",
    "evidence": "числовое обоснование",
    "based_on": ["признак1", "признак2"],
}
Как запустить
1. Клонировать репозиторий
bash
git clone <repo-url>
cd Children-Progress-Analysis
2. Создать виртуальное окружение
bash
python3 -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows
3. Установить зависимости
bash
pip install -r requirements.txt
4. Запустить Streamlit-интерфейс
bash
streamlit run app.py
Открой http://localhost:8501.

5. Или запустить CLI-пайплайн
bash
python -m src.cli
🧪 Тестирование
Запуск всех тестов:

bash
pytest
Или конкретного модуля:

bash
pytest tests/test_features.py -v
pytest tests/test_recommendations.py -v
pytest tests/test_ml.py -v
Тесты покрывают:

очистку и валидацию данных

обработку ошибок и граничных случаев

feature engineering (17 признаков)

построение и сортировку рекомендаций

сборку ML-датасета, кросс-валидацию, SHAP
