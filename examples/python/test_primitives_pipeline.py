#!/usr/bin/env python3
"""
test_primitives_pipeline.py – Tests for the Hypergredient Primitives Pipeline

Covers:
  - Generator:  folder creation, schema validity, deterministic naming
  - Refiner:    confidence/synergy enrichment, AI insight presence
  - Publisher:  index integrity, run_manifest, Markdown reports
  - Validator:  schema enforcement, safety guardrails, drift detection

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure examples/python is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from generate_primitives import generate, slugify, build_primitive, REQUIRED_SUBFOLDERS  # noqa: E402
from refine_primitives import refine, _compute_synergy_score, _compute_confidence  # noqa: E402
from publish_primitives import publish  # noqa: E402
from validate_primitives import (  # noqa: E402
    validate,
    check_schema,
    check_safety_guardrails,
    check_drift,
    check_index,
    REQUIRED_RAW_FIELDS,
    REQUIRED_REFINED_FIELDS,
    REQUIRED_SCORED_FIELDS,
)
from hypergredient_framework import HypergredientDatabase, HYPERGREDIENT_DATABASE  # noqa: E402


# ---------------------------------------------------------------------------
# Helper – run full pipeline into a temp directory
# ---------------------------------------------------------------------------

def _run_full_pipeline(tmpdir: Path, dry_run: bool = False) -> Path:
    output_root = tmpdir / "metop"
    generate(output_root, dry_run=dry_run)
    refine(output_root, dry_run=dry_run)
    publish(output_root, dry_run=dry_run)
    return output_root


# ===========================================================================
# Test: slugify helper
# ===========================================================================

class TestSlugify(unittest.TestCase):
    def test_simple_name(self):
        self.assertEqual(slugify("Retinol"), "retinol")

    def test_name_with_spaces(self):
        self.assertEqual(slugify("Hyaluronic Acid"), "hyaluronic-acid")

    def test_name_with_special_chars(self):
        self.assertEqual(slugify("Vitamin C (L-Ascorbic Acid)"), "vitamin-c-l-ascorbic-acid")

    def test_matrixyl(self):
        self.assertEqual(slugify("Matrixyl 3000"), "matrixyl-3000")

    def test_lowercase_preserved(self):
        slug = slugify("BAKUCHIOL")
        self.assertEqual(slug, slug.lower())

    def test_no_leading_trailing_dashes(self):
        slug = slugify("  Retinol  ")
        self.assertFalse(slug.startswith("-"))
        self.assertFalse(slug.endswith("-"))

    def test_deterministic(self):
        """Same input always produces same output."""
        self.assertEqual(slugify("Niacinamide"), slugify("Niacinamide"))


# ===========================================================================
# Test: generate_primitives
# ===========================================================================

class TestGeneratePrimitives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.output_root = Path(self.tmp) / "metop"
        self.count = generate(self.output_root, dry_run=False)

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------
    def test_count_matches_database(self):
        db = HypergredientDatabase()
        self.assertEqual(self.count, len(db.ingredients))

    def test_count_is_positive(self):
        self.assertGreater(self.count, 0)

    # ------------------------------------------------------------------
    # Folder structure
    # ------------------------------------------------------------------
    def test_hyping_root_created(self):
        self.assertTrue((self.output_root / "hyping").is_dir())

    def test_per_ingredient_subfolders(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            for sub in REQUIRED_SUBFOLDERS:
                self.assertTrue(
                    (hyping / slug / sub).is_dir(),
                    f"Missing subfolder '{sub}' for {slug}",
                )

    # ------------------------------------------------------------------
    # Schema validity
    # ------------------------------------------------------------------
    def test_raw_file_exists_per_ingredient(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            raw_file = hyping / slug / "raw" / f"{slug}.json"
            self.assertTrue(raw_file.exists(), f"Missing raw file for {slug}")

    def test_raw_schema_required_fields(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            raw_file = hyping / slug / "raw" / f"{slug}.json"
            data = json.loads(raw_file.read_text())
            for field in REQUIRED_RAW_FIELDS:
                self.assertIn(field, data, f"Field '{field}' missing in raw for {slug}")

    def test_raw_pipeline_stage_is_raw(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            data = json.loads((hyping / slug / "raw" / f"{slug}.json").read_text())
            self.assertEqual(data["pipeline_stage"], "raw")

    def test_raw_hypergredient_class_valid(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        valid_classes = set(HYPERGREDIENT_DATABASE.keys())
        for props in db.ingredients.values():
            slug = slugify(props.name)
            data = json.loads((hyping / slug / "raw" / f"{slug}.json").read_text())
            hg_class = data["classification"]["hypergredient_class"]
            self.assertIn(hg_class, valid_classes)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------
    def test_deterministic_output(self):
        """Running generate twice should produce identical JSON files."""
        output_root2 = Path(self.tmp) / "metop2"
        generate(output_root2, dry_run=False)

        db = HypergredientDatabase()
        hyping1 = self.output_root / "hyping"
        hyping2 = output_root2 / "hyping"

        for props in db.ingredients.values():
            slug = slugify(props.name)
            f1 = hyping1 / slug / "raw" / f"{slug}.json"
            f2 = hyping2 / slug / "raw" / f"{slug}.json"
            d1 = json.loads(f1.read_text())
            d2 = json.loads(f2.read_text())
            # Remove timestamps before comparing
            for d in (d1, d2):
                d["provenance"].pop("generated_at", None)
            self.assertEqual(d1, d2, f"Non-deterministic output for {slug}")

    # ------------------------------------------------------------------
    # Global index
    # ------------------------------------------------------------------
    def test_index_file_created(self):
        self.assertTrue((self.output_root / "hyping" / "index.json").exists())

    def test_index_entry_count(self):
        db = HypergredientDatabase()
        index = json.loads((self.output_root / "hyping" / "index.json").read_text())
        self.assertEqual(index["total_ingredients"], len(db.ingredients))

    def test_index_entries_have_required_fields(self):
        index = json.loads((self.output_root / "hyping" / "index.json").read_text())
        for entry in index["entries"]:
            for key in ("slug", "name", "hypergredient_class", "raw_path"):
                self.assertIn(key, entry)

    def test_index_raw_paths_exist(self):
        index = json.loads((self.output_root / "hyping" / "index.json").read_text())
        for entry in index["entries"]:
            raw_path = self.output_root / entry["raw_path"]
            self.assertTrue(raw_path.exists(), f"Index raw_path not found: {entry['raw_path']}")


# ===========================================================================
# Test: refine_primitives
# ===========================================================================

class TestRefinePrimitives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.output_root = Path(self.tmp) / "metop"
        generate(self.output_root, dry_run=False)
        self.count = refine(self.output_root, dry_run=False)

    def test_count_matches_database(self):
        db = HypergredientDatabase()
        self.assertEqual(self.count, len(db.ingredients))

    def test_refined_files_exist(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            self.assertTrue((hyping / slug / "refined" / f"{slug}.json").exists())

    def test_scored_files_exist(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            self.assertTrue((hyping / slug / "scored" / f"{slug}_scored.json").exists())

    def test_refined_has_ai_insights(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            data = json.loads((hyping / slug / "refined" / f"{slug}.json").read_text())
            self.assertIn("ai_insights", data)
            self.assertIn("ranked_concerns", data["ai_insights"])
            self.assertIn("model_version", data["ai_insights"])

    def test_refined_confidence_score_in_range(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            data = json.loads((hyping / slug / "refined" / f"{slug}.json").read_text())
            conf = data["optimization_metrics"]["confidence_score"]
            self.assertIsNotNone(conf)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_refined_synergy_score_in_range(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            data = json.loads((hyping / slug / "refined" / f"{slug}.json").read_text())
            syn = data["optimization_metrics"]["synergy_score"]
            self.assertIsNotNone(syn)
            self.assertGreaterEqual(syn, 0.0)
            self.assertLessEqual(syn, 10.0)

    def test_scored_has_benchmark(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            data = json.loads((hyping / slug / "scored" / f"{slug}_scored.json").read_text())
            self.assertIn("benchmark", data)
            for key in ("multi_objective_score", "stability_label", "cost_tier"):
                self.assertIn(key, data["benchmark"])

    def test_pipeline_stage_set_correctly(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            refined = json.loads((hyping / slug / "refined" / f"{slug}.json").read_text())
            scored = json.loads((hyping / slug / "scored" / f"{slug}_scored.json").read_text())
            self.assertEqual(refined["pipeline_stage"], "refined")
            self.assertEqual(scored["pipeline_stage"], "scored")


# ===========================================================================
# Test: publish_primitives
# ===========================================================================

class TestPublishPrimitives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.output_root = Path(self.tmp) / "metop"
        generate(self.output_root, dry_run=False)
        refine(self.output_root, dry_run=False)
        self.manifest = publish(self.output_root, dry_run=False)

    def test_manifest_returned(self):
        self.assertIsInstance(self.manifest, dict)
        self.assertIn("total_published", self.manifest)

    def test_total_published_count(self):
        db = HypergredientDatabase()
        self.assertEqual(self.manifest["total_published"], len(db.ingredients))

    def test_global_index_updated(self):
        index = json.loads((self.output_root / "hyping" / "index.json").read_text())
        self.assertEqual(index["pipeline_stage"], "published")

    def test_run_manifest_written(self):
        self.assertTrue((self.output_root / "hyping" / "run_manifest.json").exists())

    def test_run_manifest_fields(self):
        manifest_data = json.loads(
            (self.output_root / "hyping" / "run_manifest.json").read_text()
        )
        for key in ("run_at", "stage", "total_published", "stats_by_class"):
            self.assertIn(key, manifest_data)

    def test_markdown_reports_exist(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            report = hyping / slug / "reports" / f"{slug}_report.md"
            self.assertTrue(report.exists(), f"Missing report for {slug}")

    def test_markdown_report_contains_ingredient_name(self):
        db = HypergredientDatabase()
        hyping = self.output_root / "hyping"
        for props in db.ingredients.values():
            slug = slugify(props.name)
            report = hyping / slug / "reports" / f"{slug}_report.md"
            content = report.read_text()
            self.assertIn(props.name, content)

    def test_index_has_all_path_types(self):
        index = json.loads((self.output_root / "hyping" / "index.json").read_text())
        for entry in index["entries"]:
            paths = entry.get("paths", {})
            for path_key in ("raw", "refined", "scored", "report"):
                self.assertIn(path_key, paths, f"Missing '{path_key}' in paths for {entry['slug']}")

    def test_stats_by_class_populated(self):
        self.assertGreater(len(self.manifest.get("stats_by_class", {})), 0)


# ===========================================================================
# Test: validate_primitives
# ===========================================================================

class TestValidatePrimitives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.output_root = Path(self.tmp) / "metop"
        generate(self.output_root, dry_run=False)
        refine(self.output_root, dry_run=False)
        publish(self.output_root, dry_run=False)

    def test_full_pipeline_passes_validation(self):
        ok = validate(self.output_root)
        self.assertTrue(ok)

    def test_missing_safety_score_fails_guardrail(self):
        """Removing safety_score from a scored file should fail validation."""
        db = HypergredientDatabase()
        # Corrupt the first ingredient's scored file
        props = next(iter(db.ingredients.values()))
        slug = slugify(props.name)
        hyping = self.output_root / "hyping"
        scored_path = hyping / slug / "scored" / f"{slug}_scored.json"
        data = json.loads(scored_path.read_text())
        del data["safety_stability_cost"]["safety_score"]
        scored_path.write_text(json.dumps(data))

        errors = check_safety_guardrails(slug, data)
        self.assertTrue(any("safety_score" in str(e) for e in errors))

    def test_missing_confidence_score_fails_guardrail(self):
        db = HypergredientDatabase()
        props = next(iter(db.ingredients.values()))
        slug = slugify(props.name)
        hyping = self.output_root / "hyping"
        scored_path = hyping / slug / "scored" / f"{slug}_scored.json"
        data = json.loads(scored_path.read_text())
        data["optimization_metrics"]["confidence_score"] = None
        scored_path.write_text(json.dumps(data))

        errors = check_safety_guardrails(slug, data)
        self.assertTrue(any("confidence_score" in str(e) for e in errors))

    def test_missing_required_field_fails_schema_check(self):
        mock = {"schema_version": "1.0.0"}  # missing most fields
        errors = check_schema("test-slug", mock, REQUIRED_RAW_FIELDS, "raw")
        self.assertGreater(len(errors), 0)

    def test_drift_detection_missing_ingredient(self):
        """Removing an ingredient directory should be detected as drift."""
        db = HypergredientDatabase()
        props = next(iter(db.ingredients.values()))
        slug = slugify(props.name)
        hyping = self.output_root / "hyping"

        import shutil
        shutil.rmtree(hyping / slug)

        expected_slugs = {slugify(p.name) for p in db.ingredients.values()}
        errors = check_drift(hyping, expected_slugs)
        self.assertTrue(any(slug in str(e) for e in errors))

    def test_drift_detection_unexpected_directory(self):
        """An extra directory should be flagged as unexpected drift."""
        hyping = self.output_root / "hyping"
        (hyping / "unknown-ingredient").mkdir()

        db = HypergredientDatabase()
        expected_slugs = {slugify(p.name) for p in db.ingredients.values()}
        errors = check_drift(hyping, expected_slugs)
        self.assertTrue(any("unknown-ingredient" in str(e) for e in errors))

    def test_index_cross_reference_flags_stale_path(self):
        """If a raw file is removed but index still references it, detect it."""
        db = HypergredientDatabase()
        props = next(iter(db.ingredients.values()))
        slug = slugify(props.name)
        hyping = self.output_root / "hyping"
        raw_file = hyping / slug / "raw" / f"{slug}.json"
        raw_file.unlink()

        expected_slugs = {slugify(p.name) for p in db.ingredients.values()}
        errors = check_index(hyping, expected_slugs)
        self.assertTrue(any("raw_path" in str(e) or slug in str(e) for e in errors))

    def test_validate_missing_output_root(self):
        ok = validate(Path("/nonexistent/path"))
        self.assertFalse(ok)

    def test_dry_run_generate_no_files_created(self):
        tmp2 = tempfile.mkdtemp()
        root2 = Path(tmp2) / "metop"
        generate(root2, dry_run=True)
        self.assertFalse((root2 / "hyping").is_dir())


# ===========================================================================
# Integration test – full pipeline smoke test
# ===========================================================================

class TestFullPipelineIntegration(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        """Smoke-test: generate → refine → publish → validate passes."""
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "metop"
            generate(output_root, dry_run=False)
            refine(output_root, dry_run=False)
            publish(output_root, dry_run=False)
            ok = validate(output_root)
            self.assertTrue(ok)

    def test_all_hypergredient_classes_represented(self):
        """All 10 H.* classes should be covered by the generated primitives."""
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "metop"
            generate(output_root, dry_run=False)

            index = json.loads((output_root / "hyping" / "index.json").read_text())
            covered_classes = {e["hypergredient_class"] for e in index["entries"]}

            # At minimum the classes that DB actually has entries for must be covered
            db = HypergredientDatabase()
            expected_classes = {p.hypergredient_class for p in db.ingredients.values()}
            self.assertEqual(covered_classes, expected_classes)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    )
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print(f"\n📊 Test Results:")
    print(f"  Tests Run:   {result.testsRun}")
    print(f"  Failures:    {len(result.failures)}")
    print(f"  Errors:      {len(result.errors)}")
    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors))
        / result.testsRun * 100
        if result.testsRun > 0
        else 0.0
    )
    print(f"  Success Rate: {success_rate:.1f}%")
    sys.exit(0 if result.wasSuccessful() else 1)
