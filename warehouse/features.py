import numpy as np
import pandas as pd

CATEGORICAL = ("zone", "weekday")
RECEIVING = ("receiving_last", "receiving_mean7")
LABELS = {
    "zone": "Зона склада",
    "weekday": "День недели прогноза",
    "shipments_last": "Последний факт отгрузок",
    "shipments_lag7": "Отгрузки 7 дней назад",
    "shipments_mean7": "Средние отгрузки за неделю",
    "shipments_mean28": "Средние отгрузки за 4 недели",
    "occupancy_last": "Последний факт загрузки",
    "occupancy_lag7": "Загрузка 7 дней назад",
    "occupancy_mean7": "Средняя загрузка за неделю",
    "occupancy_mean28": "Средняя загрузка за 4 недели",
    "receiving_last": "Последний объём приёмки WMS",
    "receiving_mean7": "Средняя приёмка за неделю (WMS)",
    "year_sin": "Годовой цикл (синус)",
    "year_cos": "Годовой цикл (косинус)",
    "lead": "Горизонт, дни",
}
FEATURES = tuple(LABELS)


def build_examples(daily: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Одна строка соответствует прогнозу для зоны на дату при известном срезе origin."""
    history = daily.loc[daily["date"] <= cutoff].sort_values(["zone", "date"]).copy()
    grouped = history.groupby("zone", sort=False)
    for field in ("shipments", "occupancy", "receiving"):
        history[f"{field}_last"] = history[field]
        history[f"{field}_mean7"] = grouped[field].transform(
            lambda series: series.rolling(7, min_periods=4).mean()
        )
        if field != "receiving":
            history[f"{field}_lag7"] = grouped[field].shift(7)
            history[f"{field}_mean28"] = grouped[field].transform(
                lambda series: series.rolling(28, min_periods=14).mean()
            )
    values = history.set_index(["zone", "date"])[["shipments", "occupancy"]]
    origins = history.loc[
        history["date"] >= history["date"].min() + pd.Timedelta(days=27),
        ["date", "zone", *[name for name in FEATURES if name in history and name != "zone"]],
    ].rename(columns={"date": "origin"})

    frames = []
    for lead in range(1, horizon + 1):
        frame = origins.assign(date=origins["origin"] + pd.Timedelta(days=lead), lead=lead)
        frame["weekday"] = frame["date"].dt.dayofweek.astype(str)
        angle = 2 * np.pi * (frame["date"].dt.dayofyear - 1) / 365.25
        frame["year_sin"], frame["year_cos"] = np.sin(angle), np.cos(angle)
        for prefix, dates in (("actual", frame["date"]), ("seasonal", frame["date"] - pd.Timedelta(days=7))):
            keys = pd.MultiIndex.from_arrays([frame["zone"], dates], names=["zone", "date"])
            matched = values.reindex(keys)
            for target in values:
                frame[f"{prefix}_{target}"] = matched[target].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["origin", "zone", "lead"]).reset_index(drop=True)
