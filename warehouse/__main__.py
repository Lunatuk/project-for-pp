import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import date

from warehouse.config import load_settings
from warehouse.demo import make_demo
from warehouse.pipeline import activate_version, read_dashboard, replay, run
from warehouse.storage import connect


def main() -> int:
    parser = argparse.ArgumentParser(description="Складская логистика. Прогноз отгрузок и загрузки ячеек.")
    parser.add_argument("--config", help="Путь к TOML-конфигурации.")
    actions = parser.add_subparsers(dest="action", required=True)
    demo = actions.add_parser("demo", help="Сгенерировать историю, загрузить данные и выпустить прогноз.")
    demo.add_argument("--through", type=date.fromisoformat, help="Последняя дата синтетических данных (ГГГГ-ММ-ДД).")
    actions.add_parser("run", help="Обработать новые данные из каталогов и обновить прогноз.")
    actions.add_parser("status", help="Показать выпуск прогноза и журнал загрузки.")
    actions.add_parser("replay", help="Воспроизвести эксперимент по сохранённому снимку.").add_argument("version")
    actions.add_parser("activate", help="Активировать сохранённый выпуск или откатиться.").add_argument("version")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if args.action == "replay":
            verified = replay(settings, args.version)
            print(f"Эксперимент {verified['version'][:12]} воспроизведён для обоих показателей.")
            return 0
        if args.action == "activate":
            print(f"Активирован выпуск {activate_version(settings, args.version)[:12]}.")
            return 0
        if args.action == "status":
            with closing(connect(settings.database)) as db:
                active, diagnostics = read_dashboard(db)
            if active:
                print(f"Выпуск {active['version'][:12]}. Факт по {active['cutoff']}.")
            else:
                print("Прогноз ещё не выпущен.")
            for entry in diagnostics["runs"][:10]:
                print(f"{entry['started_at']} {entry['source']} {entry['status']}"
                      f" | файлов {entry['files_read']} | новых {entry['records_added']}"
                      f" | строк с ошибками {entry['issues']} {entry['message']}")
            return 0
        if args.action == "demo":
            make_demo(settings, args.through or settings.demo_end)
        loads, version = run(settings)
        for load in loads:
            print(f"{load.source}: {load.status} | файлов {load.files_read}"
                  f" | новых {load.records_added} | строк с ошибками {load.issues} {load.message}")
        if version is None:
            print("Один из источников недоступен. Выпуск прогноза пропущен.", file=sys.stderr)
            return 1
        print(f"Активный выпуск: {version[:12]}")
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
