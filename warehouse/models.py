from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from warehouse.config import Settings
from warehouse.features import CATEGORICAL, FEATURES, RECEIVING

TARGETS = {
    "shipments": ("Отгрузки", "палет"),
    "occupancy": ("Загрузка ячеек", "п. п."),
}
NAMES = {
    "seasonal": "Наивная (тот же день недели)",
    "ridge": "Линейная Ridge",
    "boosting": "Градиентный бустинг",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    profile: int = 0

    @property
    def uses_receiving(self) -> bool:
        return all(name in self.features for name in RECEIVING)

    @property
    def params(self) -> dict:
        if self.name == "ridge":
            return {"alpha": 10}
        if self.name == "boosting":
            leaves, rate = BOOSTING_PROFILES[self.profile]
            return {"max_iter": 125, "max_leaf_nodes": leaves, "learning_rate": rate,
                    "l2_regularization": 1.0, "early_stopping": False}
        return {}

    def specification(self) -> dict:
        return {"model": self.name, "receiving": self.uses_receiving, "profile": self.profile}

    @classmethod
    def from_specification(cls, spec: dict) -> "Candidate":
        features = FEATURES if spec["receiving"] else tuple(name for name in FEATURES if name not in RECEIVING)
        return cls(spec["model"], () if spec["model"] == "seasonal" else features, spec["profile"])


BOOSTING_PROFILES = ((15, 0.08), (7, 0.09), (31, 0.06), (15, 0.04))


def candidates(budget: int, incumbent: Candidate | None) -> list[Candidate]:
    variants = [Candidate("seasonal", ())]
    for name, profiles in (("ridge", range(1)), ("boosting", range(budget))):
        for profile in profiles:
            for receiving in (True, False):
                variants.append(Candidate.from_specification({"model": name, "receiving": receiving, "profile": profile}))
    if incumbent is not None and incumbent not in variants:
        variants.append(incumbent)
    return variants


def select_candidate(comparison: list[tuple[Candidate, dict]], incumbent: Candidate | None, threshold: float):
    best = min(comparison, key=lambda item: item[1]["mae"])
    if incumbent is None:
        return best, {"action": "initial", "best": best[0].specification()}
    previous = next(item for item in comparison if item[0] == incumbent)
    error = previous[1]["mae"]
    improvement = (error - best[1]["mae"]) / error if error else 0.0
    changed = best[0] != incumbent and improvement > 0 and improvement >= threshold
    return (best if changed else previous), {
        "action": "promoted" if changed else "retained",
        "incumbent": incumbent.specification(), "best": best[0].specification(),
        "relative_improvement": improvement, "threshold": threshold,
    }


def clip(values: np.ndarray, target: str) -> np.ndarray:
    return np.clip(values, 0, 100 if target == "occupancy" else None)


@dataclass
class Predictor:
    candidate: Candidate
    target: str
    estimator: object

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.candidate.name == "seasonal":
            values = (
                rows[f"seasonal_{self.target}"]
                .fillna(rows[f"{self.target}_last"])
                .fillna(rows[f"{self.target}_mean7"])
                .fillna(float(self.estimator))
                .to_numpy()
            )
        else:
            with threadpool_limits(limits=2):
                values = self.estimator.predict(rows[list(self.candidate.features)])
        return clip(values, self.target)


def fit(candidate: Candidate, target: str, rows: pd.DataFrame, seed: int) -> Predictor:
    actual = rows[f"actual_{target}"]
    if rows.empty:
        raise ValueError(f"Не хватает обучающих значений для {TARGETS[target][0].lower()}.")
    if candidate.name == "seasonal":
        return Predictor(candidate, target, float(actual.median()))
    categorical = [name for name in CATEGORICAL if name in candidate.features]
    numeric = [name for name in candidate.features if name not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    if candidate.name == "ridge":
        regressor = Ridge(**candidate.params)
    else:
        regressor = HistGradientBoostingRegressor(**candidate.params, random_state=seed)
    pipeline = make_pipeline(preprocessor, regressor)
    with threadpool_limits(limits=2):
        pipeline.fit(rows[list(candidate.features)], actual)
    return Predictor(candidate, target, pipeline)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = np.asarray(predicted) - np.asarray(actual)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "n": len(errors),
    }


def splits(cutoff: pd.Timestamp, settings: Settings) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], pd.Timestamp]:
    # На правой границе отложенного периода доступны факты по всем горизонтам.
    stop = cutoff - pd.Timedelta(days=settings.horizon - 1)
    holdout = stop - pd.Timedelta(days=settings.holdout_days)
    days = pd.Timedelta(days=settings.validation_days)
    cv_stop = holdout - pd.Timedelta(days=settings.horizon)
    folds = [(cv_stop - days * i, cv_stop - days * (i - 1)) for i in (3, 2, 1)]
    return folds, holdout


def partition(rows: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Обучение ограничено датой известности ФАКТА, а не датой формирования признаков.
    return rows.loc[rows["date"] < start], rows.loc[rows["origin"].between(start, stop, inclusive="left")]


def serialize_points(rows: pd.DataFrame, values: np.ndarray) -> list[dict]:
    points = rows[["origin", "date", "zone", "lead"]].copy()
    for field in ("origin", "date"):
        points[field] = points[field].dt.strftime("%Y-%m-%d")
    points["forecast"] = values.round(4)
    return points.to_dict("records")


def evaluate(
    examples: pd.DataFrame, cutoff: pd.Timestamp, settings: Settings, target: str,
    incumbent: Candidate | None = None,
) -> tuple[dict, dict]:
    actual = f"actual_{target}"
    rows = examples.dropna(subset=[actual])
    future = examples.loc[examples["origin"] == cutoff]
    folds, holdout_start = splits(cutoff, settings)
    holdout_stop = cutoff - pd.Timedelta(days=settings.horizon - 1)
    training, holdout = partition(rows, holdout_start, holdout_stop)
    validation = [partition(rows, start, stop) for start, stop in folds]
    if future.empty or holdout.empty or any(len(train) < 100 or test.empty for train, test in validation):
        raise ValueError("Слишком короткая история для трёх временных окон и отложенной проверки.")

    comparison = []
    for candidate in candidates(settings.search_budget, incumbent):
        folding = []
        observed, predictions = [], []
        for (start, stop), (train, test) in zip(folds, validation):
            guess = fit(candidate, target, train, settings.seed).predict(test)
            observed.extend(test[actual])
            predictions.extend(guess)
            folding.append({
                "from": start.date().isoformat(), "until_exclusive": stop.date().isoformat(),
                **metrics(test[actual], guess),
            })
        comparison.append((candidate, {
            **candidate.specification(), "params": candidate.params,
            **metrics(observed, predictions), "folds": folding,
        }))

    # Выбор признаков и алгоритма выполняется до обращения к отложенной выборке.
    selected = {
        name: min((item for item in comparison if item[0].name == name), key=lambda entry: entry[1]["mae"])
        for name in NAMES
    }
    (winner, winner_score), promotion = select_candidate(comparison, incumbent, settings.promotion_threshold)
    selected[winner.name] = (winner, winner_score)
    holdout_scores = []
    for candidate, _ in selected.values():
        model = fit(candidate, target, training, settings.seed)
        values = model.predict(holdout)
        holdout_scores.append({
            **candidate.specification(), "params": candidate.params, **metrics(holdout[actual], values),
        })
        if candidate == winner:
            checked, checked_values = model, values

    production = fit(winner, target, rows, settings.seed)
    points = serialize_points(holdout, checked_values)
    for point, observed in zip(points, holdout[actual]):
        point["actual"] = round(observed, 4)
    result = {
        "winner": winner.name,
        "selection": winner.specification(),
        "promotion": promotion,
        "receiving": winner.uses_receiving,
        "features": list(winner.features),
        "cv": [score for _, score in comparison],
        "holdout": holdout_scores,
        "holdout_from": holdout_start.date().isoformat(),
        "holdout_until": (holdout_stop - pd.Timedelta(days=1)).date().isoformat(),
        "checked": points,
        "forecast": serialize_points(future, production.predict(future)),
    }
    return result, {"checked": checked, "production": production}
