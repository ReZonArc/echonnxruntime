#!/usr/bin/env python3
"""
generate_primitives.py – Stage 1: Raw Hypergredient Primitive Generator

Extracts raw hypergredient primitives from HypergredientDatabase and
MetaOptimizationEngine outputs, writing one JSON file per ingredient
into the categorised output tree:

  metop/hyping/<ingredient-slug>/raw/<ingredient-slug>.json

A global index is written to:
  metop/hyping/index.json

Usage:
  python generate_primitives.py [--output-root <path>] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Resolve paths so the script works both from repo root and from this dir
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from hypergredient_framework import HypergredientDatabase, HYPERGREDIENT_DATABASE  # noqa: E402
from meta_optimization_engine import MetaOptimizationEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"
PIPELINE_STAGE = "raw"
REQUIRED_SUBFOLDERS = ("raw", "refined", "scored", "reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Return a deterministic, filesystem-safe slug for an ingredient name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


def build_primitive(ingredient_key: str, props, run_ts: float) -> Dict[str, Any]:
    """Build the canonical JSON primitive for one ingredient."""
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_stage": PIPELINE_STAGE,
        "provenance": {
            "source_module": "hypergredient_framework.HypergredientDatabase",
            "ingredient_key": ingredient_key,
            "generated_at": run_ts,
            "generator": "generate_primitives.py",
        },
        "identity": {
            "name": props.name,
            "inci_name": props.inci_name,
            "slug": slugify(props.name),
        },
        "classification": {
            "hypergredient_class": props.hypergredient_class,
            "class_description": HYPERGREDIENT_DATABASE.get(props.hypergredient_class, ""),
            "primary_function": props.primary_function,
            "secondary_functions": sorted(props.secondary_functions),
        },
        "optimization_metrics": {
            "efficacy_score": props.efficacy_score,
            "bioavailability": props.bioavailability,
            "synergy_score": None,          # filled by refine stage
            "confidence_score": None,       # filled by refine stage
        },
        "safety_stability_cost": {
            "safety_score": props.safety_score,
            "clinical_evidence": props.clinical_evidence,
            "pH_min": props.pH_min,
            "pH_max": props.pH_max,
            "stability_conditions": props.stability_conditions,
            "cost_per_gram_zar": props.cost_per_gram,
        },
        "relationships": {
            "synergies": sorted(props.synergies),
            "incompatibilities": sorted(props.incompatibilities),
        },
    }


def ensure_ingredient_tree(base: Path, slug: str, dry_run: bool) -> Path:
    """Create the per-ingredient subfolder tree and return ingredient root."""
    ingredient_root = base / slug
    for sub in REQUIRED_SUBFOLDERS:
        target = ingredient_root / sub
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
            # place .gitkeep so empty dirs survive git
            gk = target / ".gitkeep"
            if not gk.exists():
                gk.write_text("")
    return ingredient_root


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate(output_root: Path, dry_run: bool) -> int:
    """
    Generate raw primitives for all ingredients in HypergredientDatabase.

    Returns the number of primitives written (or that would be written).
    """
    run_ts = time.time()
    db = HypergredientDatabase()

    # Also warm up MetaOptimizationEngine to ensure consistent output
    engine = MetaOptimizationEngine()
    _ = engine.optimization_space  # trigger space init

    hyping_root = output_root / "hyping"
    if not dry_run:
        hyping_root.mkdir(parents=True, exist_ok=True)

    index_entries: List[Dict[str, Any]] = []
    count = 0

    for ingredient_key, props in sorted(db.ingredients.items()):
        slug = slugify(props.name)
        primitive = build_primitive(ingredient_key, props, run_ts)

        ingredient_root = ensure_ingredient_tree(hyping_root, slug, dry_run)
        raw_file = ingredient_root / "raw" / f"{slug}.json"

        if not dry_run:
            raw_file.write_text(json.dumps(primitive, indent=2, sort_keys=True))

        index_entries.append({
            "ingredient_key": ingredient_key,
            "slug": slug,
            "name": props.name,
            "hypergredient_class": props.hypergredient_class,
            "raw_path": str(raw_file.relative_to(output_root)),
        })

        count += 1
        status = "[DRY-RUN]" if dry_run else "✓"
        print(f"  {status} {slug}  →  {props.hypergredient_class}")

    # Write global index
    index = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_stage": PIPELINE_STAGE,
        "generated_at": run_ts,
        "generator": "generate_primitives.py",
        "total_ingredients": count,
        "hypergredient_classes": sorted(HYPERGREDIENT_DATABASE.keys()),
        "entries": sorted(index_entries, key=lambda e: e["slug"]),
    }
    index_file = hyping_root / "index.json"
    if not dry_run:
        index_file.write_text(json.dumps(index, indent=2, sort_keys=True))
        print(f"\n📋 Index written → {index_file}")
    else:
        print(f"\n[DRY-RUN] Would write index → {index_file}")

    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate raw hypergredient primitives for coschem meta-optimization."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "metop",
        help="Root directory for primitive output tree (default: <repo>/metop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating any files.",
    )
    args = parser.parse_args()

    print("🧬 Hypergredient Primitive Generator – Stage 1: raw")
    print(f"   Output root : {args.output_root}")
    print(f"   Dry-run     : {args.dry_run}\n")

    count = generate(args.output_root, args.dry_run)

    print(f"\n✅ Done – {count} primitives {'would be ' if args.dry_run else ''}generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
