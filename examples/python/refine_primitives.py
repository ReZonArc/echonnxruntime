#!/usr/bin/env python3
"""
refine_primitives.py – Stage 2: Hypergredient Primitive Refinement

Reads raw primitives produced by generate_primitives.py, enriches them
using FormulationEvolution and HypergredientAI scoring hooks, and writes
refined output to:

  metop/hyping/<ingredient-slug>/refined/<ingredient-slug>.json
  metop/hyping/<ingredient-slug>/scored/<ingredient-slug>_scored.json

Usage:
  python refine_primitives.py [--output-root <path>] [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from hypergredient_framework import (  # noqa: E402
    HypergredientDatabase,
    HypergredientFormulator,
    HypergredientProperties,
    FormulationRequest,
    HYPERGREDIENT_DATABASE,
)
from hypergredient_advanced import (  # noqa: E402
    HypergredientAI,
    FormulationFeedback,
    FormulationEvolution,
)
from generate_primitives import slugify  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"
PIPELINE_STAGE_REFINED = "refined"
PIPELINE_STAGE_SCORED = "scored"


# ---------------------------------------------------------------------------
# Refinement helpers
# ---------------------------------------------------------------------------

def _load_raw(raw_path: Path) -> Optional[Dict[str, Any]]:
    """Load raw primitive JSON; return None on error."""
    if not raw_path.exists():
        return None
    try:
        return json.loads(raw_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _compute_synergy_score(props: HypergredientProperties, db: HypergredientDatabase) -> float:
    """
    Compute a normalised synergy score for an ingredient.

    Score is based on the ingredient scoring utility applied to each synergistic
    partner already in a hypothetical formulation context, then averaged.
    """
    if not props.synergies:
        return 5.0  # neutral baseline

    objective_weights = {
        "efficacy": 0.35,
        "safety": 0.25,
        "stability": 0.20,
        "cost": 0.15,
        "synergy": 0.05,
    }

    scores = []
    for partner_key in props.synergies:
        partner = db.ingredients.get(partner_key)
        if partner:
            s = db.calculate_ingredient_score(
                partner, objective_weights, formulation_context=[props.name.lower()]
            )
            scores.append(s)

    if not scores:
        return 5.0

    # Scale to 0-10
    avg = sum(scores) / len(scores)
    return round(min(10.0, avg * 10.0), 4)


def _compute_confidence(props: HypergredientProperties, synergy_score: float) -> float:
    """
    Estimate a confidence score in [0, 1] for the primitive.

    Combines efficacy, safety, clinical evidence strength, and synergy.
    """
    evidence_map = {"Strong": 1.0, "Moderate": 0.7, "Weak": 0.4, "None": 0.1}
    evidence_weight = evidence_map.get(props.clinical_evidence, 0.3)

    raw_confidence = (
        0.35 * (props.efficacy_score / 10.0)
        + 0.25 * (props.safety_score / 10.0)
        + 0.20 * evidence_weight
        + 0.10 * props.bioavailability
        + 0.10 * (synergy_score / 10.0)
    )
    return round(min(1.0, raw_confidence), 4)


def _ai_ranked_concerns(props: HypergredientProperties) -> List[str]:
    """
    Use HypergredientAI to predict optimal concerns for this ingredient.
    Returns a ranked list of concern strings.
    """
    ai = HypergredientAI()
    concerns: List[str] = [props.primary_function.replace("_", " ")]
    for fn in props.secondary_functions:
        concerns.append(fn.replace("_", " "))

    request = FormulationRequest(
        target_concerns=concerns[:3],  # limit to 3 to avoid noise
        skin_type="normal",
        budget=1500.0,
        preferences=["stable"],
    )
    predictions = ai.predict_optimal_combination(request)
    return [p[0] for p in predictions[:5]]


def refine_one(
    raw: Dict[str, Any],
    props: HypergredientProperties,
    db: HypergredientDatabase,
    run_ts: float,
) -> Dict[str, Any]:
    """Return a refined primitive enriched with scored metrics."""
    synergy_score = _compute_synergy_score(props, db)
    confidence = _compute_confidence(props, synergy_score)
    ai_concerns = _ai_ranked_concerns(props)

    refined = dict(raw)
    refined["schema_version"] = SCHEMA_VERSION
    refined["pipeline_stage"] = PIPELINE_STAGE_REFINED
    refined["provenance"] = {
        **raw["provenance"],
        "refined_at": run_ts,
        "refiner": "refine_primitives.py",
        "source_stage": "raw",
    }
    refined["optimization_metrics"] = {
        **raw["optimization_metrics"],
        "synergy_score": synergy_score,
        "confidence_score": confidence,
    }
    refined["ai_insights"] = {
        "ranked_concerns": ai_concerns,
        "model_version": HypergredientAI().model_version,
    }
    return refined


def score_one(
    refined: Dict[str, Any],
    props: HypergredientProperties,
    db: HypergredientDatabase,
    run_ts: float,
) -> Dict[str, Any]:
    """Produce a scored overlay on top of the refined primitive."""
    objective_weights = {
        "efficacy": 0.35,
        "safety": 0.25,
        "stability": 0.20,
        "cost": 0.15,
        "synergy": 0.05,
    }
    base_score = db.calculate_ingredient_score(props, objective_weights)
    stability_flag = props.stability_conditions.get("stable", False)
    stability_label = "stable" if stability_flag else "sensitive"

    scored = dict(refined)
    scored["schema_version"] = SCHEMA_VERSION
    scored["pipeline_stage"] = PIPELINE_STAGE_SCORED
    scored["provenance"] = {
        **refined["provenance"],
        "scored_at": run_ts,
        "scorer": "refine_primitives.py",
        "source_stage": "refined",
    }
    scored["benchmark"] = {
        "multi_objective_score": round(base_score, 4),
        "stability_label": stability_label,
        "ph_range_width": round(props.pH_max - props.pH_min, 2),
        "cost_tier": (
            "budget" if props.cost_per_gram < 50
            else "mid" if props.cost_per_gram < 200
            else "premium"
        ),
    }
    return scored


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

def refine(output_root: Path, dry_run: bool) -> int:
    run_ts = time.time()
    db = HypergredientDatabase()
    hyping_root = output_root / "hyping"

    if not hyping_root.is_dir():
        print(
            f"❌ hyping root not found: {hyping_root}\n"
            "   Run generate_primitives.py first.",
            file=sys.stderr,
        )
        return 0

    count = 0
    for ingredient_key, props in sorted(db.ingredients.items()):
        slug = slugify(props.name)
        ingredient_root = hyping_root / slug
        raw_path = ingredient_root / "raw" / f"{slug}.json"
        refined_path = ingredient_root / "refined" / f"{slug}.json"
        scored_path = ingredient_root / "scored" / f"{slug}_scored.json"

        raw = _load_raw(raw_path)
        if raw is None:
            print(f"  ⚠  Skipping {slug}: raw primitive not found at {raw_path}")
            continue

        refined = refine_one(raw, props, db, run_ts)
        scored = score_one(refined, props, db, run_ts)

        if not dry_run:
            refined_path.parent.mkdir(parents=True, exist_ok=True)
            scored_path.parent.mkdir(parents=True, exist_ok=True)
            refined_path.write_text(json.dumps(refined, indent=2, sort_keys=True))
            scored_path.write_text(json.dumps(scored, indent=2, sort_keys=True))

        status = "[DRY-RUN]" if dry_run else "✓"
        conf = refined["optimization_metrics"]["confidence_score"]
        syn = refined["optimization_metrics"]["synergy_score"]
        print(f"  {status} {slug}  confidence={conf:.3f}  synergy={syn:.1f}")
        count += 1

    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine raw hypergredient primitives using AI scoring hooks."
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

    print("🔮 Hypergredient Primitive Refiner – Stage 2: refined + scored")
    print(f"   Output root : {args.output_root}")
    print(f"   Dry-run     : {args.dry_run}\n")

    count = refine(args.output_root, args.dry_run)

    print(f"\n✅ Done – {count} primitives {'would be ' if args.dry_run else ''}refined.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
