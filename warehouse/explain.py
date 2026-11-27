import numpy as np
import pandas as pd

from warehouse.features import CATEGORICAL, LABELS
from warehouse.models import Predictor


def feature_names(model: Predictor) -> tuple[str, ...]:
    if model.candidate.name == "seasonal":
        target = model.target
        return f"seasonal_{target}", f"{target}_last", f"{target}_mean7"
    return model.candidate.features


def feature_label(name: str) -> str:
    return {
        "seasonal_shipments": "Отгрузки в тот же день прошлой недели",
        "seasonal_occupancy": "Загрузка в тот же день прошлой недели",
    }.get(name, LABELS.get(name, name))


def reference_row(training: pd.DataFrame, model: Predictor) -> dict:
    reference = {}
    for name in feature_names(model):
        values = training[name].dropna()
        if values.empty:
            reference[name] = np.nan
        elif name in CATEGORICAL:
            reference[name] = values.mode().iloc[0]
        else:
            reference[name] = values.median()
    return reference


def contributions(
    model: Predictor, rows: pd.DataFrame, reference: dict, seed: int, permutations: int = 16
) -> tuple[float, pd.DataFrame]:
    """Разложение прогноза по траекториям от опорной строки к объясняемой."""
    features = feature_names(model)
    rows = rows.reset_index(drop=True)
    origin = pd.DataFrame([reference] * len(rows), columns=features)
    rng = np.random.default_rng(seed)
    paths = []
    orders = []
    for _ in range(permutations):
        state = origin.copy()
        paths.append(state)
        order = rng.permutation(features)
        orders.append(order)
        for name in order:
            state = state.assign(**{name: rows[name]})
            paths.append(state)
    predictions = model.predict(pd.concat(paths, ignore_index=True)).reshape(
        permutations, len(features) + 1, len(rows)
    )
    effects = pd.DataFrame(0.0, index=rows.index, columns=features)
    for order, path in zip(orders, predictions):
        for name, change in zip(order, np.diff(path, axis=0)):
            effects[name] += change / permutations
    return float(predictions[0, 0, 0]), effects


def importance(model: Predictor, rows: pd.DataFrame, reference: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actual = rows[f"actual_{model.target}"].to_numpy()
    base_error = np.mean(np.abs(actual - model.predict(rows)))
    permutation = {}
    for name in feature_names(model):
        errors = []
        for _ in range(4):
            changed = rows.assign(**{name: rng.permutation(rows[name].to_numpy())})
            errors.append(np.mean(np.abs(actual - model.predict(changed))))
        permutation[name] = np.mean(errors) - base_error
    sample = rows.sample(min(100, len(rows)), random_state=seed)
    _, effects = contributions(model, sample, reference, seed)
    return pd.DataFrame({"permutation": permutation, "local": effects.abs().mean()})


def partial_dependence(model: Predictor, rows: pd.DataFrame, feature: str) -> pd.DataFrame:
    values = rows[feature].dropna()
    if values.empty:
        return pd.DataFrame(columns=("value", "forecast"))
    grid = np.unique(values.quantile(np.linspace(0.05, 0.95, 12)).to_numpy())
    sample = rows.sample(min(300, len(rows)), random_state=0)
    variants = pd.concat([sample.assign(**{feature: value}) for value in grid], ignore_index=True)
    prediction = model.predict(variants).reshape(len(grid), len(sample)).mean(axis=1)
    return pd.DataFrame({"value": grid, "forecast": prediction})
