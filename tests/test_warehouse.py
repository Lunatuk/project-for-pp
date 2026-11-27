import json
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from warehouse.config import load_settings
from warehouse.demo import make_demo
from warehouse.explain import contributions
from warehouse.features import FEATURES, build_examples
from warehouse.ingest import ingest, save_feed
from warehouse.mart import prepare
from warehouse.models import TARGETS, Candidate, partition, select_candidate, splits
from warehouse.monitoring import population_stability
from warehouse.pipeline import activate_version, load_artifacts, read_control, read_dashboard, replay, run
from warehouse.storage import connect


class WarehouseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        self.settings = replace(
            load_settings(Path(__file__).resolve().parents[1] / "config.toml"),
            database=folder / "warehouse.sqlite3", shipments=folder / "carrier", wms_events=folder / "wms",
            start=date(2025, 1, 1), demo_end=date(2025, 11, 19), validation_days=14, holdout_days=21,
            search_budget=1,
        )

    def test_ingest_retries_isolates_faults_and_avoids_duplicates(self):
        settings = self.settings
        make_demo(settings, settings.demo_end)
        corrupt = settings.wms_events / "corrupt.jsonl"
        corrupt.write_text('{"type":\n', encoding="utf-8")
        with closing(connect(settings.database)) as db:
            for load in ingest(db, settings):
                self.assertGreater(load.records_added, 0)
            self.assertGreater(db.execute("SELECT COUNT(*) FROM import_issues").fetchone()[0], 0)
            unchanged = ingest(db, settings)
            self.assertEqual([(x.files_read, x.records_added) for x in unchanged], [(0, 0), (0, 0)])
            errors = db.execute("SELECT COUNT(*) FROM import_issues").fetchone()[0]
            with corrupt.open("a", encoding="utf-8") as file:
                file.write("\n")
            self.assertEqual(ingest(db, settings)[1].issues, 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM import_issues").fetchone()[0], errors)
            payload = b"id,date,zone,pallets\n"
            self.assertEqual(save_feed(settings.shipments, ".csv", payload),
                             save_feed(settings.shipments, ".csv", payload))

            appended_day = settings.demo_end + timedelta(days=1)
            make_demo(settings, appended_day)
            make_demo(settings, settings.demo_end)
            extended = ingest(db, settings)
            self.assertEqual([x.records_added for x in extended], [3, 6])
            settings.wms_events.rename(settings.wms_events.with_name("offline"))
            self.assertEqual([x.status for x in ingest(db, settings)], ["ok", "error"])

    def test_missing_facts_do_not_become_zero(self):
        settings = self.settings
        make_demo(settings, settings.demo_end)
        with closing(connect(settings.database)) as db:
            ingest(db, settings)
            with db:
                daily, cutoff = prepare(db, settings)
            self.assertEqual(cutoff.date(), settings.demo_end)
            missing = daily.loc[(daily["date"] == pd.Timestamp("2025-02-11")) & (daily["zone"] == "A")].iloc[0]
            self.assertTrue(pd.isna(missing["shipments"]))
            self.assertTrue(pd.notna(missing["occupancy"]))
            invalid = daily.loc[(daily["date"] == pd.Timestamp("2025-03-06")) & (daily["zone"] == "C")].iloc[0]
            self.assertTrue(pd.isna(invalid["shipments"]))
            self.assertIsNotNone(db.execute("""
                SELECT 1 FROM quality_issues WHERE zone='C' AND date='2025-03-06' AND indicator='shipments'
            """).fetchone())

    def test_features_are_as_of_and_partitions_never_share_labels(self):
        cutoff = pd.Timestamp("2025-11-19")
        daily = pd.DataFrame({
            "date": pd.date_range(self.settings.start, cutoff), "zone": "A",
            "shipments": np.arange(323, dtype=float), "occupancy": 75.0, "receiving": 60.0,
        })
        original = build_examples(daily, cutoff, 7)
        origin = pd.Timestamp("2025-08-01")
        altered = daily.copy()
        altered.loc[altered["date"] > origin, ["shipments", "occupancy", "receiving"]] += 10_000
        updated = build_examples(altered, cutoff, 7)
        used = [*FEATURES, "seasonal_shipments", "seasonal_occupancy"]
        mask = original["origin"] <= origin
        pd.testing.assert_frame_equal(original.loc[mask, used], updated.loc[mask, used])
        self.assertFalse(original.loc[mask, "actual_shipments"].equals(updated.loc[mask, "actual_shipments"]))

        folds, holdout = splits(cutoff, self.settings)
        known = original.dropna(subset=["actual_shipments"])
        for start, end in [*folds, (holdout, cutoff - pd.Timedelta(days=6))]:
            train, test = partition(known, start, end)
            self.assertLess(train["date"].max(), test["origin"].min())
        self.assertLess(folds[-1][1] + pd.Timedelta(days=6), holdout)

    def test_pipeline_reuses_release_and_preserves_it_during_outage(self):
        settings = self.settings
        make_demo(settings, settings.demo_end)
        _, version = run(settings)
        self.assertIsNotNone(version)
        unchanged, repeated_version = run(settings)
        self.assertEqual(version, repeated_version)
        self.assertEqual([x.records_added for x in unchanged], [0, 0])

        with closing(connect(settings.database)) as db:
            release, _ = read_dashboard(db)
            payload = json.loads(release["payload"])
            models = load_artifacts(db, version)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0], 1)
            for target in TARGETS:
                report = payload["targets"][target]
                self.assertEqual(len(report["holdout"]), 3)
                self.assertEqual(len(report["forecast"]), 7 * len(settings.zones))
                self.assertEqual({point["lead"] for point in report["forecast"]}, set(range(1, 8)))
                fit = models[target]
                row = fit["future_rows"].iloc[:1]
                baseline, effects = contributions(fit["production"], row, fit["production_reference"], settings.seed)
                self.assertAlmostEqual(baseline + effects.iloc[0].sum(), fit["production"].predict(row)[0])
            self.assertTrue(all(0 <= p["forecast"] <= 100 for p in payload["targets"]["occupancy"]["forecast"]))

        settings.wms_events.rename(settings.wms_events.with_name("offline"))
        loads, blocked = run(settings)
        self.assertIsNone(blocked)
        self.assertEqual(loads[1].status, "error")
        with closing(connect(settings.database)) as db:
            active, _ = read_dashboard(db)
            self.assertEqual(active["version"], version)

        settings.wms_events.with_name("offline").rename(settings.wms_events)
        next_day = settings.demo_end + timedelta(days=1)
        make_demo(settings, next_day)
        loads, updated = run(settings)
        self.assertNotEqual(updated, version)
        self.assertEqual([x.records_added for x in loads], [3, 6])
        with closing(connect(settings.database)) as db:
            active, _ = read_dashboard(db)
            self.assertEqual(active["version"], updated)
            self.assertEqual(active["cutoff"], next_day.isoformat())
            control = read_control(db)
            self.assertEqual(control["versions"][0]["version"], updated)
            self.assertGreater(control["versions"][0]["mae"]["shipments"], 0)
            check = control["checks"][0]
            self.assertEqual(check["performance"]["shipments"]["n"], 3)
            self.assertIsNotNone(check["performance"]["shipments"]["mae"])
            self.assertIsNotNone(check["performance"]["shipments"]["reference_mae"])

        self.assertEqual(replay(settings, version[:12])["targets"], list(TARGETS))
        self.assertEqual(activate_version(settings, version[:12]), version)
        _, displayed = run(settings)
        self.assertEqual(displayed, version)
        with closing(connect(settings.database)) as db:
            self.assertEqual(read_control(db)["activations"][0]["action"], "manual")
        self.assertEqual(activate_version(settings, updated[:12]), updated)

    def test_monitor_detects_shift_and_promotion_requires_improvement(self):
        base = pd.Series(np.linspace(70, 80, num=105))
        self.assertAlmostEqual(population_stability(base, base), 0)
        self.assertGreater(population_stability(base, base + 20), 0.2)
        self.assertIsNone(population_stability(base, pd.Series([np.nan])))
        incumbent = Candidate("seasonal", ())
        challenger = Candidate("ridge", FEATURES)
        result, decision = select_candidate([(incumbent, {"mae": 10}), (challenger, {"mae": 9.95})], incumbent, 0.01)
        self.assertEqual(result[0], incumbent)
        self.assertEqual(decision["action"], "retained")
        result, decision = select_candidate([(incumbent, {"mae": 10}), (challenger, {"mae": 9.5})], incumbent, 0.01)
        self.assertEqual(result[0], challenger)
        self.assertEqual(decision["action"], "promoted")


if __name__ == "__main__":
    unittest.main()
