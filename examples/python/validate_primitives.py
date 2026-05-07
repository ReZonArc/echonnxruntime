#!/usr/bin/env python3
"""
validate_primitives.py – Schema validation, drift checks, and safety guardrails

Validates the hypergredient primitive tree produced by the generation and
refinement pipeline.  Exits with code 0 on success, 1 on any failure.

Checks:
  1. Required subfolders exist per ingredient.
  2. Each primitive JSON has all required schema fields.
  3. Safety guardrails: safety_score and confidence_score are present and
     within acceptable ranges.
  4. Drift detection: expected ingredient slugs (from HypergredientDatabase)
     are all present in the tree.
  5. Index cross-reference: index.json entries match the files on disk.

Usage:
  python validate_primitives.py [--output-root <path>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from hypergredient_framework import HypergredientDatabase  # noqa: E402
from generate_primitives import slugify  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_SUBFOLDERS = {"raw", "refined", "scored", "reports"}
REQUIRED_RAW_FIELDS = {
    "schema_version",
    "pipeline_stage",
    "provenance",
    "identity",
    "classification",
    "optimization_metrics",
    "safety_stability_cost",
    "relationships",
}
REQUIRED_REFINED_FIELDS = REQUIRED_RAW_FIELDS | {"ai_insights"}
REQUIRED_SCORED_FIELDS = REQUIRED_REFINED_FIELDS | {"benchmark"}

MIN_SAFETY_SCORE = 0.0
MAX_SAFETY_SCORE = 10.0
MIN_CONFIDENCE_SCORE = 0.0
MAX_CONFIDENCE_SCORE = 1.0

# Minimum acceptable values (guardrails)
GUARDRAIL_MIN_SAFETY = 1.0    # safety_score must be at least 1.0
GUARDRAIL_MIN_CONFIDENCE = 0.0  # confidence_score must be present (any value ≥ 0)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationError:
    def __init__(self, slug: str, check: str, detail: str):
        self.slug = slug
        self.check = check
        self.detail = detail

    def __str__(self) -> str:
        return f"[{self.check}] {self.slug}: {self.detail}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc


def check_folder_structure(hyping_root: Path, slug: str) -> List[ValidationError]:
    errors: List[ValidationError] = []
    ingredient_root = hyping_root / slug
    if not ingredient_root.is_dir():
        errors.append(ValidationError(slug, "folder_structure", f"directory missing: {ingredient_root}"))
        return errors

    for sub in REQUIRED_SUBFOLDERS:
        sub_dir = ingredient_root / sub
        if not sub_dir.is_dir():
            errors.append(
                ValidationError(slug, "folder_structure", f"required subfolder missing: {sub}")
            )
    return errors


def check_schema(slug: str, data: Dict[str, Any], required_fields: Set[str], stage: str) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if data is None:
        errors.append(ValidationError(slug, f"schema_{stage}", f"{stage} file is missing or unreadable"))
        return errors

    for field in sorted(required_fields):
        if field not in data:
            errors.append(ValidationError(slug, f"schema_{stage}", f"missing required field: '{field}'"))
    return errors


def check_safety_guardrails(slug: str, scored: Dict[str, Any]) -> List[ValidationError]:
    errors: List[ValidationError] = []
    if scored is None:
        return errors

    safety_score = scored.get("safety_stability_cost", {}).get("safety_score")
    confidence_score = scored.get("optimization_metrics", {}).get("confidence_score")

    if safety_score is None:
        errors.append(ValidationError(slug, "safety_guardrail", "safety_score is None (missing)"))
    elif not (MIN_SAFETY_SCORE <= safety_score <= MAX_SAFETY_SCORE):
        errors.append(
            ValidationError(slug, "safety_guardrail", f"safety_score {safety_score} out of range [0, 10]")
        )
    elif safety_score < GUARDRAIL_MIN_SAFETY:
        errors.append(
            ValidationError(slug, "safety_guardrail", f"safety_score {safety_score} below minimum guardrail ({GUARDRAIL_MIN_SAFETY})")
        )

    if confidence_score is None:
        errors.append(ValidationError(slug, "safety_guardrail", "confidence_score is None (missing)"))
    elif not (MIN_CONFIDENCE_SCORE <= confidence_score <= MAX_CONFIDENCE_SCORE):
        errors.append(
            ValidationError(slug, "safety_guardrail", f"confidence_score {confidence_score} out of range [0, 1]")
        )
    return errors


def check_drift(hyping_root: Path, expected_slugs: Set[str]) -> List[ValidationError]:
    """Detect added/deleted ingredients vs what the DB declares."""
    errors: List[ValidationError] = []
    present_slugs: Set[str] = set()

    for d in hyping_root.iterdir():
        if d.is_dir():
            present_slugs.add(d.name)

    missing = expected_slugs - present_slugs
    unexpected = present_slugs - expected_slugs

    for slug in sorted(missing):
        errors.append(ValidationError(slug, "drift_check", "expected ingredient directory is missing"))

    for slug in sorted(unexpected):
        errors.append(ValidationError(slug, "drift_check", "unexpected directory found (not in DB)"))

    return errors


def check_index(hyping_root: Path, expected_slugs: Set[str]) -> List[ValidationError]:
    """Cross-reference index.json against on-disk files."""
    errors: List[ValidationError] = []
    index_path = hyping_root / "index.json"

    if not index_path.exists():
        errors.append(ValidationError("index", "index_check", "index.json is missing"))
        return errors

    try:
        index = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(ValidationError("index", "index_check", f"index.json is unreadable: {exc}"))
        return errors

    indexed_slugs = {e["slug"] for e in index.get("entries", [])}
    for slug in sorted(expected_slugs - indexed_slugs):
        errors.append(ValidationError(slug, "index_check", "slug present on disk but missing from index.json"))
    for slug in sorted(indexed_slugs - expected_slugs):
        errors.append(ValidationError(slug, "index_check", "slug in index.json but not in DB"))

    # Validate each entry's raw_path points to existing file
    for entry in index.get("entries", []):
        raw_rel = entry.get("raw_path") or entry.get("paths", {}).get("raw")
        if raw_rel:
            full_path = hyping_root.parent / raw_rel
            if not full_path.exists():
                errors.append(
                    ValidationError(entry.get("slug", "?"), "index_check", f"raw_path does not exist: {raw_rel}")
                )
    return errors


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def validate(output_root: Path) -> bool:
    hyping_root = output_root / "hyping"

    if not hyping_root.is_dir():
        print(f"❌ hyping root not found: {hyping_root}", file=sys.stderr)
        return False

    db = HypergredientDatabase()
    expected_slugs: Set[str] = {slugify(p.name) for p in db.ingredients.values()}

    all_errors: List[ValidationError] = []

    # 1. Drift detection
    drift_errors = check_drift(hyping_root, expected_slugs)
    all_errors.extend(drift_errors)

    # 2. Index cross-reference
    index_errors = check_index(hyping_root, expected_slugs)
    all_errors.extend(index_errors)

    # 3. Per-ingredient checks
    present_slugs = {d.name for d in hyping_root.iterdir() if d.is_dir()}
    check_slugs = expected_slugs & present_slugs  # only check what's present

    counts = {"checked": 0, "raw_ok": 0, "refined_ok": 0, "scored_ok": 0, "report_ok": 0}

    for slug in sorted(check_slugs):
        ingredient_root = hyping_root / slug
        counts["checked"] += 1

        # Folder structure
        all_errors.extend(check_folder_structure(hyping_root, slug))

        # Raw schema
        raw_path = ingredient_root / "raw" / f"{slug}.json"
        try:
            raw = _load_json(raw_path)
        except ValueError as exc:
            all_errors.append(ValidationError(slug, "schema_raw", str(exc)))
            raw = None

        raw_errors = check_schema(slug, raw, REQUIRED_RAW_FIELDS, "raw")
        all_errors.extend(raw_errors)
        if not raw_errors:
            counts["raw_ok"] += 1

        # Refined schema
        refined_path = ingredient_root / "refined" / f"{slug}.json"
        try:
            refined = _load_json(refined_path)
        except ValueError as exc:
            all_errors.append(ValidationError(slug, "schema_refined", str(exc)))
            refined = None

        refined_errors = check_schema(slug, refined, REQUIRED_REFINED_FIELDS, "refined")
        all_errors.extend(refined_errors)
        if not refined_errors and refined is not None:
            counts["refined_ok"] += 1

        # Scored schema + safety guardrails
        scored_path = ingredient_root / "scored" / f"{slug}_scored.json"
        try:
            scored = _load_json(scored_path)
        except ValueError as exc:
            all_errors.append(ValidationError(slug, "schema_scored", str(exc)))
            scored = None

        scored_errors = check_schema(slug, scored, REQUIRED_SCORED_FIELDS, "scored")
        all_errors.extend(scored_errors)
        guardrail_errors = check_safety_guardrails(slug, scored)
        all_errors.extend(guardrail_errors)
        if not scored_errors and not guardrail_errors and scored is not None:
            counts["scored_ok"] += 1

        # Report presence
        report_path = ingredient_root / "reports" / f"{slug}_report.md"
        if report_path.exists():
            counts["report_ok"] += 1

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔍 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Ingredients checked : {counts['checked']}")
    print(f"  Raw    OK           : {counts['raw_ok']}")
    print(f"  Refined OK          : {counts['refined_ok']}")
    print(f"  Scored OK           : {counts['scored_ok']}")
    print(f"  Reports present     : {counts['report_ok']}")
    print(f"  Drift errors        : {len(drift_errors)}")
    print(f"  Index errors        : {len(index_errors)}")
    print(f"  Total errors        : {len(all_errors)}")
    print("=" * 60)

    if all_errors:
        print("\n❌ ERRORS FOUND:\n")
        for err in all_errors:
            print(f"  {err}")
        return False

    print("\n✅ All validations passed.")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate hypergredient primitives tree for schema, safety, and drift."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "metop",
        help="Root directory for primitive output tree (default: <repo>/metop)",
    )
    args = parser.parse_args()

    print("🔍 Hypergredient Primitive Validator")
    print(f"   Output root : {args.output_root}\n")

    ok = validate(args.output_root)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
