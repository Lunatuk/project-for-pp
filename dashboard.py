import json
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_controls import data_actions, demo_available, monitor_page, process, versions_page
from warehouse.config import load_settings
from warehouse.explain import contributions, feature_label, feature_names, partial_dependence
from warehouse.features import CATEGORICAL, FEATURES
from warehouse.models import NAMES, TARGETS, metrics
from warehouse.pipeline import load_artifacts, read_control, read_dashboard

ACTUAL = "#344761"
FORECAST = "#007f78"
ALERT = "#ad5730"
PAGES = ("Прогноз", "Модели и контроль", "Данные")
MODEL_PAGES = ("Качество моделей", "Объяснения", "Мониторинг", "Эксперименты и версии")


def chart_layout(figure: go.Figure, ylabel: str, height: int = 350) -> go.Figure:
    return figure.update_layout(
        height=height, margin=dict(l=15, r=15, t=20, b=15), yaxis_title=ylabel,
        legend=dict(orientation="h", x=0, y=1.12), hovermode="x unified",
    )


def forecast_page(payload: dict, report: dict, target: str, zone: str, days: int, alert: float) -> None:
    occupancy = target == "occupancy"
    unit = "%" if occupancy else "палет"
    history = pd.DataFrame(payload["history"])
    history = history.loc[history["zone"] == zone].tail(days)
    future = pd.DataFrame(report["forecast"]).query("zone == @zone").sort_values("date")
    if future.empty:
        st.info("Для этой зоны ещё нет опубликованного прогноза.")
        return
    latest = history.iloc[-1]
    peak = future.loc[future["forecast"].idxmax()]
    last, next_days, busiest = st.columns(3)
    last.metric("Последний факт", f"{latest[target]:.0f} {unit}", help=f"На {latest['date']}")
    if occupancy:
        next_days.metric("Дней у порога или выше", str(int(future["forecast"].ge(alert).sum())),
                         help=f"Порог загрузки {alert:g}%")
    else:
        next_days.metric("Отгрузки за горизонт", f"{future['forecast'].sum():.0f} палет")
    busiest.metric("Пик прогноза", f"{peak['forecast']:.0f} {unit}",
                   help=datetime.fromisoformat(peak["date"]).strftime("%d.%m.%Y"))

    figure = go.Figure()
    figure.add_scatter(x=history["date"], y=history[target], name="Факт", line=dict(color=ACTUAL, width=2))
    figure.add_scatter(x=future["date"], y=future["forecast"], name="Прогноз", mode="lines+markers",
                       line=dict(color=FORECAST, width=3, dash="dash"), marker=dict(size=8))
    if occupancy:
        figure.add_hline(y=alert, line_dash="dot", line_color=ALERT,
                         annotation_text=f"Порог {alert:g}%")
        figure.update_yaxes(range=[0, 105], ticksuffix="%")
    else:
        figure.update_yaxes(rangemode="tozero")
    figure.update_xaxes(type="date")
    st.plotly_chart(chart_layout(figure, TARGETS[target][0] + f", {unit}"), use_container_width=True)

    if occupancy:
        dates = pd.to_datetime(future.loc[future["forecast"] >= alert, "date"]).dt.strftime("%d.%m").tolist()
        if dates:
            st.warning(f"Загрузка у порога {alert:g}% или выше ожидается {', '.join(dates)}. "
                       "Проверьте свободные ячейки и план размещения.")
        else:
            st.success(f"На всём горизонте прогноз загрузки ниже {alert:g}%.")
    else:
        peak_date = datetime.fromisoformat(peak["date"]).strftime("%d.%m")
        st.info(f"Наибольший объём ожидается {peak_date}. "
                f"Плановый ориентир для зоны {zone} — {peak['forecast']:.0f} палет.")
    daily_forecast = future[["date", "forecast"]].copy()
    daily_forecast["forecast"] = daily_forecast["forecast"].round(1 if occupancy else 0)
    st.dataframe(daily_forecast.rename(columns={"date": "Дата", "forecast": f"Прогноз, {unit}"}),
                 hide_index=True, use_container_width=True)
    score = next(item for item in report["holdout"] if item["model"] == report["winner"])
    st.caption(f"{NAMES[report['winner']]} · средняя абсолютная ошибка на отложенном периоде "
               f"{score['mae']:.1f} {TARGETS[target][1]} по всем зонам и горизонтам.")


def model_table(scores: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Модель": NAMES[row["model"]], "Данные приёмки": "Да" if row["receiving"] else "Нет",
            "Параметры": (
                f"листья {row['params']['max_leaf_nodes']}, шаг {row['params']['learning_rate']}"
                if "max_leaf_nodes" in row.get("params", {}) else ""
            ),
            "MAE": round(row["mae"], 2), "RMSE": round(row["rmse"], 2), "Оценок": row["n"],
        }
        for row in scores
    ])


def quality_page(report: dict, target: str, zone: str, horizon: int) -> None:
    unit = TARGETS[target][1]
    st.subheader("Ошибка прогноза на отложенном периоде")
    st.caption(f"Даты выпуска прогноза {report['holdout_from']} — {report['holdout_until']}. "
               f"MAE и RMSE выражены в единицах показателя, {unit}.")
    st.dataframe(model_table(report["holdout"]), hide_index=True, use_container_width=True)
    score = next(item for item in report["holdout"] if item["model"] == report["winner"])
    base = next(item for item in report["holdout"] if item["model"] == "seasonal")
    difference = base["mae"] - score["mae"]
    st.write(
        f"Средняя ошибка выбранной модели — **{score['mae']:.2f} {unit}**. "
        f"У прогноза по тому же дню прошлой недели — **{base['mae']:.2f} {unit}**. "
        f"Разница MAE (базовая минус выбранная) — {difference:+.2f} {unit}."
    )

    lead = st.selectbox("Горизонт для проверки в выбранной зоне", range(1, horizon + 1),
                        format_func=lambda value: f"{value} дн.")
    points = pd.DataFrame(report["checked"]).query("zone == @zone and lead == @lead").sort_values("date")
    if points.empty:
        st.info("Для выбранного разреза нет подтверждённых фактов.")
    else:
        local = metrics(points["actual"], points["forecast"])
        figure = go.Figure()
        figure.add_scatter(x=points["date"], y=points["actual"], name="Факт", line=dict(color=ACTUAL))
        figure.add_scatter(x=points["date"], y=points["forecast"], name="Прогноз",
                           line=dict(color=FORECAST, dash="dash"))
        st.plotly_chart(chart_layout(figure, TARGETS[target][0]), use_container_width=True)
        st.caption(f"Зона {zone}, горизонт {lead} дн. · MAE {local['mae']:.2f} {unit} · "
                   f"RMSE {local['rmse']:.2f} {unit} · {local['n']} прогнозов с фактом.")

    with st.expander("Выбор модели и проверка признаков приёмки"):
        st.write(f"Активная конфигурация для этого выпуска — «{NAMES[report['winner']]}». "
                 "Кандидаты сравнивались на трёх одинаковых окнах валидации.")
        decision = report.get("promotion", {})
        if decision.get("action") in ("retained", "promoted"):
            change = "Конфигурация обновлена." if decision["action"] == "promoted" else "Действующая конфигурация сохранена."
            st.caption(f"{change} Улучшение лучшего кандидата по MAE — {decision['relative_improvement']:.2%}. "
                       f"Порог публикации конфигурации — {decision['threshold']:.2%}.")
        st.dataframe(model_table(report["cv"]), hide_index=True, use_container_width=True)
        details = [
            {**row, **fold}
            for row in report["cv"] for fold in row["folds"]
        ]
        table = model_table(details)
        table.insert(0, "Окно [с, до)", [f"{row['from']} — {row['until_exclusive']}" for row in details])
        st.dataframe(table, hide_index=True, use_container_width=True)


@st.cache_resource(show_spinner=False)
def version_artifacts(uri: str, version: str) -> dict:
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return load_artifacts(db, version)


def explanation_page(report: dict, target: str, zone: str, artifacts: dict, seed: int) -> None:
    st.subheader("Что повлияло на отдельный прогноз")
    period = st.radio("Прогнозы для объяснения", ("Будущие даты", "Отложенный период"), horizontal=True)
    future = period == "Будущие даты"
    key = "production" if future else "checked"
    model = artifacts[key]
    rows = artifacts["future_rows" if future else "held_rows"].query("zone == @zone")
    if rows.empty:
        st.info("Для этой зоны нет прогнозов в выбранном периоде.")
    else:
        days = sorted(rows["date"].unique())
        day = st.selectbox("Дата прогноза", days, format_func=lambda value: pd.Timestamp(value).strftime("%d.%m.%Y"))
        dated = rows.loc[rows["date"] == day]
        lead = st.selectbox("Горизонт объясняемого прогноза", sorted(dated["lead"].unique()))
        row = dated.loc[dated["lead"] == lead]
        baseline, effects = contributions(model, row, artifacts[f"{key}_reference"], seed)
        prediction = model.predict(row)[0]
        baseline_card, prediction_card = st.columns(2)
        baseline_card.metric("Прогноз для опорной строки", f"{baseline:.2f}")
        prediction_card.metric("Прогноз для выбранной даты", f"{prediction:.2f}")
        contributions_table = []
        for name, change in effects.iloc[0].items():
            value = row[name].iloc[0]
            if name == "weekday" and pd.notna(value):
                value = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")[int(value)]
            elif isinstance(value, float):
                value = f"{value:.2f}" if pd.notna(value) else "Пропуск"
            contributions_table.append({
                "Признак": feature_label(name), "Значение": str(value), "Вклад": change,
                "Размер": abs(change),
            })
        table = pd.DataFrame(contributions_table).sort_values("Размер", ascending=False).drop(columns="Размер")
        st.dataframe(table, column_config={"Вклад": st.column_config.NumberColumn(format="%+.2f")},
                     hide_index=True, use_container_width=True)
        st.caption(f"Прогноз выпущен {row['origin'].iloc[0]:%d.%m.%Y}. "
                   "Прогноз для опорной строки и сумма вкладов дают итоговое значение.")

    st.subheader("Какие признаки важны в разные периоды")
    ranking = pd.DataFrame(report["importance"])
    if ranking.empty:
        return
    method = st.radio("Метод оценки", ("Перемешивание признака", "Средний абсолютный вклад"), horizontal=True)
    column = "permutation" if method == "Перемешивание признака" else "local"
    ranking["Признак"] = ranking["feature"].map(feature_label)
    table = ranking.pivot(index="Признак", columns="period", values=column)
    order = table.abs().mean(axis=1).sort_values(ascending=False).index
    st.dataframe(table.loc[order].round(3), use_container_width=True)
    if column == "permutation":
        st.caption("Рост MAE после перемешивания показывает, насколько модель опирается на признак.")
    else:
        st.caption("Показан средний модуль вклада относительно опорной строки на каждом периоде.")

    st.subheader("Как меняется прогноз при изменении признака")
    numeric = [name for name in feature_names(artifacts["checked"]) if name not in CATEGORICAL]
    feature = st.selectbox("Признак", numeric, format_func=feature_label)
    curve = partial_dependence(artifacts["checked"], artifacts["held_rows"], feature)
    if curve.empty:
        st.info("Нет измеренных значений этого признака.")
    else:
        figure = px.line(curve, x="value", y="forecast", markers=True, color_discrete_sequence=[FORECAST])
        figure.update_xaxes(title=feature_label(feature))
        st.plotly_chart(chart_layout(figure, "Средний прогноз", 280), use_container_width=True)
        st.caption("Модель пересчитывает прогнозы для отложенного периода при заданном значении признака. "
                   "Полученная зависимость описывает поведение модели.")


def sources_page(diagnostics: dict, zone: str) -> None:
    st.subheader("Журнал загрузок")
    if diagnostics["runs"]:
        logs = pd.DataFrame(diagnostics["runs"]).drop(columns="id")
        logs["source"] = logs["source"].map({"carrier": "Отгрузки CSV", "wms": "События WMS JSONL"})
        logs = logs.rename(columns={
            "started_at": "Начало (UTC)", "source": "Источник", "status": "Статус", "files_read": "Файлов",
            "records_added": "Новых записей", "issues": "Ошибок строк", "message": "Диагностика",
        })
        st.dataframe(logs, hide_index=True, use_container_width=True)
    else:
        st.info("Записи появятся после первого запуска загрузчика.")
    st.caption(f"Число строк в карантине — {diagnostics['quarantined']}.")
    st.subheader(f"Проверки суточных данных · зона {zone}")
    issues = [row for row in diagnostics["quality"] if row["zone"] == zone]
    if issues:
        frame = pd.DataFrame(issues)
        frame["indicator"] = frame["indicator"].map({
            "shipments": "Отгрузки", "occupancy": "Загрузка ячеек", "receiving": "Приёмка WMS",
        })
        st.dataframe(frame.rename(columns={
            "date": "Дата", "zone": "Зона", "indicator": "Показатель", "reason": "Причина",
        }), hide_index=True, use_container_width=True)
    else:
        st.info("Для этой зоны нет замечаний к суточным данным.")


def main() -> None:
    st.set_page_config(page_title="Складская логистика", layout="wide")
    try:
        settings = load_settings()
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"Ошибка конфигурации. {exc}")
        return
    st.title("Складская логистика")
    st.sidebar.header("Разрез данных")
    zone = st.sidebar.selectbox("Зона", sorted(settings.zones))
    target = st.sidebar.radio("Показатель", TARGETS.keys(), format_func=lambda name: TARGETS[name][0])
    days = st.sidebar.selectbox("Глубина факта", (30, 60, 90, 180), index=1, format_func=lambda n: f"{n} дней")
    page = st.radio("Раздел", PAGES, horizontal=True, label_visibility="collapsed")
    section = st.radio("Раздел моделей", MODEL_PAGES, horizontal=True) if page == "Модели и контроль" else None
    if notice := st.session_state.pop("warehouse_notice", None):
        st.success(notice)
    if error := st.session_state.pop("warehouse_error", None):
        st.error(error)
    uri = settings.database.as_uri() + "?mode=ro"

    @st.fragment(run_every="15s")
    def content() -> None:
        if not settings.database.is_file():
            data_actions(settings, False)
            return
        try:
            with closing(sqlite3.connect(uri, uri=True)) as db:
                db.execute("BEGIN")
                active, diagnostics = read_dashboard(db)
                control = read_control(db) if section in ("Мониторинг", "Эксперименты и версии") else None
        except sqlite3.Error as exc:
            st.error(f"База данных временно недоступна. {exc}")
            return
        if active:
            cutoff = datetime.fromisoformat(active["cutoff"]).strftime("%d.%m.%Y")
            st.caption(f"Факт по {cutoff} · зона {zone} · {settings.zones[zone]} ячеек · "
                       f"выпуск {active['version'][:12]}")
        if page == "Данные":
            data_actions(settings, bool(active))
            sources_page(diagnostics, zone)
            if active:
                st.subheader(f"Витрина признаков · зона {zone}")
                sample = version_artifacts(uri, active["version"])[target]["future_rows"].query("zone == @zone")
                columns = ["origin", "date", *FEATURES]
                names = {"origin": "Дата выпуска", "date": "Дата прогноза", **{name: feature_label(name) for name in FEATURES}}
                st.dataframe(sample[columns].rename(columns=names), hide_index=True, use_container_width=True)
            return
        if not active:
            st.info("Для прогноза нужны закрытые суточные данные обоих источников.")
            data_actions(settings, False)
            return
        latest_runs = {name: next((r for r in diagnostics["runs"] if r["source"] == name), None)
                       for name in ("carrier", "wms")}
        if any(row and row["status"] == "error" for row in latest_runs.values()):
            st.warning("При последней загрузке один из источников был недоступен. Показан сохранённый выпуск.")
        elif diagnostics["loaded_through"] and diagnostics["loaded_through"] > active["cutoff"]:
            st.warning("В базе есть более новые данные. Проверьте полноту фактов или выберите актуальный выпуск.")
        payload = json.loads(active["payload"])
        report = payload["targets"][target]
        horizon = max(row["lead"] for row in report["forecast"])
        if page == "Прогноз":
            forecast_page(payload, report, target, zone, days, settings.occupancy_alert)
        elif section == "Качество моделей":
            quality_page(report, target, zone, horizon)
        elif section == "Объяснения":
            artifacts = version_artifacts(uri, active["version"])
            explanation_page(report, target, zone, artifacts[target], settings.seed)
        elif section == "Мониторинг":
            monitor_page(control, settings)
        elif section == "Эксперименты и версии":
            versions_page(control, active["version"], settings)

    content()


if __name__ == "__main__":
    main()
