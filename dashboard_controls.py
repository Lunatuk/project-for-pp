import json
import sqlite3

import pandas as pd
import streamlit as st

from warehouse.config import Settings
from warehouse.demo import make_demo
from warehouse.ingest import save_feed
from warehouse.models import NAMES, TARGETS
from warehouse.pipeline import activate_version, replay, run

INDICATOR_NAMES = {
    "shipments": "Отгрузки", "occupancy": "Загрузка ячеек", "receiving": "Приёмка WMS",
}


def process(settings: Settings, demo: bool = False, files: tuple = ()) -> None:
    try:
        with st.spinner("Загружаем факты и проверяем модели…"):
            if demo:
                make_demo(settings, settings.demo_end)
            for folder, suffix, content in files:
                save_feed(folder, suffix, content)
            loads, version = run(settings)
        if version is None:
            st.session_state["warehouse_error"] = "Нужны данные обоих источников. Причины указаны в журнале загрузок."
        else:
            count = sum(item.records_added for item in loads)
            st.session_state["warehouse_notice"] = f"Новых записей — {count}. Активный выпуск {version[:12]}."
        st.rerun()
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f"Обработка приостановлена. {exc}")


def demo_available(settings: Settings) -> bool:
    return not any(settings.shipments.glob("*.csv")) and not any(settings.wms_events.glob("*.jsonl"))


def data_actions(settings: Settings, active: bool) -> None:
    st.subheader("Загрузка данных")
    if not active and demo_available(settings) and st.button("Создать демоданные", type="primary"):
        process(settings, demo=True)
    with st.form("warehouse_upload", clear_on_submit=True):
        carrier = st.file_uploader("Суточные реестры отгрузок", type=["csv"], accept_multiple_files=True)
        wms = st.file_uploader("Журналы WMS", type=["jsonl"], accept_multiple_files=True)
        st.caption("Файлы загружаются в текущую базу. Для отдельной истории укажите другой WAREHOUSE_CONFIG.")
        if st.form_submit_button("Загрузить и обновить прогноз"):
            files = tuple(
                (folder, suffix, item.getvalue())
                for folder, suffix, items in ((settings.shipments, ".csv", carrier), (settings.wms_events, ".jsonl", wms))
                for item in items
            )
            process(settings, files=files)


def monitor_page(control: dict, settings: Settings) -> None:
    st.subheader("Мониторинг дрейфа и ошибок")
    if not control["checks"]:
        st.info("Проверка появится после обновления прогноза.")
    else:
        latest = control["checks"][0]
        status = "Рекомендуется переобучение" if latest["recommend"] else "Пороги мониторинга соблюдены"
        (st.warning if latest["recommend"] else st.success)(status)
        st.caption(f"Данные с {latest['current_from']} по {latest['current_until']}. "
                   f"Опорное окно с {latest['reference_from']} по {latest['reference_until']}. "
                   f"Выпуск {latest['version'][:12]}.")
        drift, errors = st.columns(2)
        with drift:
            st.markdown("**Распределение входных показателей**")
            st.dataframe(pd.DataFrame([
                {"Ряд": INDICATOR_NAMES[name], "PSI": score} for name, score in latest["psi"].items()
            ]).round(3), hide_index=True, use_container_width=True)
        with errors:
            st.markdown("**Ошибка ранее выпущенных прогнозов**")
            st.dataframe(pd.DataFrame([
                {"Показатель": TARGETS[name][0], "MAE по новым фактам": score["mae"],
                 "Опорная MAE": score["reference_mae"], "Фактов": score["n"]}
                for name, score in latest["performance"].items()
            ]).round(2), hide_index=True, use_container_width=True)
        st.caption(f"Сигнал срабатывает при PSI ≥ {latest['drift_threshold']:g} или росте MAE более чем на "
                   f"{latest['error_growth_threshold']:.0%} при наличии хотя бы {latest['monitor_min_points']} "
                   "сопоставленных прогнозов. Оцениваются прогнозы, выпущенные до появления факта.")
        with st.expander("История проверок"):
            st.dataframe(pd.DataFrame([
                {"Выпуск": row["version"][:12], "По дату": row["current_until"],
                 "Сигнал": "Есть" if row["recommend"] else "Нет",
                 **{f"PSI · {INDICATOR_NAMES[name]}": value for name, value in row["psi"].items()}}
                for row in control["checks"]
            ]).round(3), hide_index=True, use_container_width=True)

    if st.button("Проверить новые данные и переобучить"):
        process(settings)
    st.caption("Новая конфигурация заменяет действующую при достаточном снижении MAE на одинаковых окнах валидации. "
               "При поступлении данных действующая конфигурация также обучается на обновлённой истории.")


def versions_page(control: dict, active: str, settings: Settings) -> None:
    st.subheader("Журнал экспериментов и версии")
    st.dataframe(pd.DataFrame([
        {"Выпуск": row["version"][:12], "Факт по": row["cutoff"], "Активен": row["version"] == active,
         "Отгрузки": NAMES[row["shipments"]], "Загрузка": NAMES[row["occupancy"]],
         "MAE, палеты": row["mae"]["shipments"], "MAE, п. п.": row["mae"]["occupancy"],
         "Зерно": row["experiment"]["settings"]["seed"] if row["experiment"] else None}
        for row in control["versions"]
    ]).round(2), hide_index=True, use_container_width=True)
    versions = {row["version"]: row for row in control["versions"]}
    selected = st.selectbox("Выпуск", versions,
                            format_func=lambda key: f"{key[:12]} · {versions[key]['cutoff']}")
    details = versions[selected]["experiment"]
    if details:
        st.caption(f"SHA-256 данных — {details['checksum']}")
        with st.expander("Параметры эксперимента"):
            st.code(json.dumps(details["settings"], ensure_ascii=False, indent=2), language="json")
        if st.button("Воспроизвести эксперимент"):
            try:
                with st.spinner("Повторяем поиск на сохранённом снимке…"):
                    result = replay(settings, selected)
                st.success(f"Метрики и выбор моделей совпали для выпуска {result['version'][:12]}.")
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(str(exc))
    else:
        st.caption("Выпуск сохранён до добавления журнала экспериментов.")
    if selected != active and st.button("Активировать выбранный выпуск"):
        try:
            version = activate_version(settings, selected)
            st.session_state["warehouse_notice"] = f"Активирован выпуск {version[:12]}."
            st.rerun()
        except (OSError, sqlite3.Error, ValueError) as exc:
            st.error(str(exc))
    if control["activations"]:
        with st.expander("Журнал активаций"):
            st.dataframe(pd.DataFrame([
                {"Время (UTC)": row["at"], "Выпуск": row["version"][:12],
                 "Действие": {"publish": "Публикация", "manual": "Ручная активация"}[row["action"]]}
                for row in control["activations"]
            ]), hide_index=True, use_container_width=True)
