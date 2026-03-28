#!/usr/bin/env python3
"""
DSFS Training Script — Train the India friction predictor on REAL data.

Data Sources (not synthetic!):
  - World Bank Open API  → GDP, unemployment, inflation, poverty, Gini (real India values)
  - ACLED Conflict Data  → Real India protest/riot events as y-labels (2000-2024)
  - CMIE, RBI, MOSPI    → Curated India socioeconomic indicators
  - V-Dem               → Political stability & press freedom

Run: python scripts/train_predictor.py
Colab: python scripts/train_predictor.py --colab --augment 100
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.predictor.core import DSFSPredictor
from src.predictor.data.real_data_pipeline import IndiaRealDataPipeline


def main():
    parser = argparse.ArgumentParser(description="Train DSFS on real India data")
    parser.add_argument("--augment", type=int, default=50,
                        help="Augmentation factor per real data point (default: 50)")
    parser.add_argument("--colab", action="store_true",
                        help="Use higher augmentation for Colab training")
    parser.add_argument("--api", action="store_true",
                        help="Fetch live data from World Bank API (needs internet)")
    parser.add_argument("--acled-key", type=str, default=None,
                        help="ACLED API key for live conflict data")
    args = parser.parse_args()

    # Colab uses more augmentation for better generalization
    augment_factor = 100 if args.colab else args.augment

    print("=" * 70)
    print("  DSFS — Dynamic Society Friction Simulator (India)")
    print("  Real Data Training Pipeline v2.0")
    print("=" * 70)
    print(f"\n  Data mode: {'World Bank API (live)' if args.api else 'Curated historical (offline)'}")
    print(f"  ACLED conflict data: {'API (live)' if args.acled_key else 'Curated (22 real events)'}")
    print(f"  Augmentation factor: {augment_factor}x per real data point")

    # ─── Step 1: Build Real Training Dataset ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 1: Loading Real India Data (2000-2024)")
    print("─" * 70)

    pipeline = IndiaRealDataPipeline(
        acled_api_key=args.acled_key,
        use_api=args.api
    )

    X, y_risk, y_severity = pipeline.build_training_dataset(augment_factor=augment_factor)

    # Show dataset summary
    summary = pipeline.get_dataset_summary()
    print(f"\n  Data Sources:")
    for ref in summary["references"]:
        print(f"    • {ref}")
    print(f"\n  Dataset: {len(X)} samples from {summary['years_covered']}")
    print(f"  Real India years: {summary['total_real_years']}")
    print(f"  Conflict events used: {summary['conflict_events']}")
    print(f"  Distribution:")
    print(f"    LOW     (0-40%):  {np.sum(y_severity == 0)} samples")
    print(f"    MEDIUM  (40-60%): {np.sum(y_severity == 1)} samples")
    print(f"    HIGH    (60-80%): {np.sum(y_severity == 2)} samples")
    print(f"    CRITICAL(80%+):   {np.sum(y_severity == 3)} samples")

    # ─── Step 2: Train the LGBM Model ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 2: Training LGBM Risk Predictor on Real Data")
    print("─" * 70)

    predictor = DSFSPredictor(model_dir="models")
    predictor.lgbm.train(X, y_risk, y_severity)
    predictor.lgbm.save()
    predictor.is_trained = True

    print("  ✅ LGBM trained and saved to models/lgbm/")

    # ─── Step 3: India-Specific Validation Tests ───────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 3: India Validation Tests (6 scenarios)")
    print("─" * 70)

    test_cases = [
        {
            "name": "Gujarat Riots 2002",
            "indicators": {
                "unemployment_rate": 4.5, "inflation_rate": 4.3, "gini_coefficient": 0.33,
                "youth_bulge_pct": 46.0, "political_stability": 0.05, "press_freedom_index": 0.35,
                "gdp_growth": 3.9, "poverty_rate": 43.0, "urbanization_rate": 28.1,
                "internet_penetration": 1.0, "ethnic_fractionalization": 0.55,
                "corruption_index": 27, "military_expenditure_pct": 2.7, "food_price_index": 88.0
            },
            "expected_level": "CRITICAL"
        },
        {
            "name": "Farmer Protests 2020-21",
            "indicators": {
                "unemployment_rate": 8.8, "inflation_rate": 6.2, "gini_coefficient": 0.37,
                "youth_bulge_pct": 49.7, "political_stability": 0.10, "press_freedom_index": 0.36,
                "gdp_growth": -6.6, "poverty_rate": 22.0, "urbanization_rate": 35.4,
                "internet_penetration": 55.0, "ethnic_fractionalization": 0.42,
                "corruption_index": 40, "military_expenditure_pct": 2.9, "food_price_index": 130.0
            },
            "expected_level": "CRITICAL"
        },
        {
            "name": "Agnipath Protests 2022",
            "indicators": {
                "unemployment_rate": 7.3, "inflation_rate": 6.7, "gini_coefficient": 0.36,
                "youth_bulge_pct": 50.0, "political_stability": 0.10, "press_freedom_index": 0.34,
                "gdp_growth": 7.2, "poverty_rate": 19.0, "urbanization_rate": 36.4,
                "internet_penetration": 63.0, "ethnic_fractionalization": 0.42,
                "corruption_index": 40, "military_expenditure_pct": 2.4, "food_price_index": 140.0
            },
            "expected_level": "HIGH"
        },
        {
            "name": "Manipur Violence 2023",
            "indicators": {
                "unemployment_rate": 12.0, "inflation_rate": 5.4, "gini_coefficient": 0.42,
                "youth_bulge_pct": 52.0, "political_stability": 0.08, "press_freedom_index": 0.28,
                "gdp_growth": 1.5, "poverty_rate": 28.0, "urbanization_rate": 30.0,
                "internet_penetration": 35.0, "ethnic_fractionalization": 0.65,
                "corruption_index": 30, "military_expenditure_pct": 3.5, "food_price_index": 145.0
            },
            "expected_level": "CRITICAL"
        },
        {
            "name": "India 2024 Current",
            "indicators": pipeline.get_current_india_indicators(),
            "expected_level": "HIGH"
        },
        {
            "name": "India Stable (2004-05 Boom)",
            "indicators": {
                "unemployment_rate": 4.4, "inflation_rate": 3.8, "gini_coefficient": 0.33,
                "youth_bulge_pct": 46.5, "political_stability": 0.28, "press_freedom_index": 0.55,
                "gdp_growth": 7.9, "poverty_rate": 35.0, "urbanization_rate": 29.0,
                "internet_penetration": 2.0, "ethnic_fractionalization": 0.35,
                "corruption_index": 32, "military_expenditure_pct": 2.5, "food_price_index": 90.0
            },
            "expected_level": "MEDIUM"
        }
    ]

    all_passed = True
    for tc in test_cases:
        result = predictor.predict_quick(tc["indicators"])
        risk = result["risk_score"]
        level = result["risk_level"]
        passed = level == tc["expected_level"]
        status = "✅" if passed else "⚠️"
        if not passed:
            all_passed = False
        print(f"\n  {status} {tc['name']}")
        print(f"     Risk: {risk:.1f}%  |  Level: {level}  |  Expected: {tc['expected_level']}")

    # ─── Step 4: What-If Validation ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 4: What-If Policy Validation")
    print("─" * 70)

    farmer_indicators = test_cases[1]["indicators"]
    base_risk = predictor.predict_quick(farmer_indicators)["risk_score"]

    interventions_to_test = [
        ("pmgkay_activation", "PMGKAY Free Ration"),
        ("expand_mgnrega", "MGNREGA Expansion"),
        ("internet_shutdown", "Internet Shutdown (bad policy)")
    ]

    for int_id, int_name in interventions_to_test:
        whatif = predictor.what_if(farmer_indicators, int_id, base_risk)
        change = whatif["risk_change"]
        direction = "↓" if change < 0 else "↑"
        print(f"\n  {int_name}:")
        print(f"    {whatif['risk_before']}% → {whatif['risk_after']}% "
              f"({direction}{abs(change):.1f} pts) | {whatif['level_before']} → {whatif['level_after']}")

    # ─── Step 5: State Risk Comparison ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 5: India State-wise Risk Comparison")
    print("─" * 70)

    states = ["Punjab", "Manipur", "Bihar", "Haryana", "Gujarat", "Maharashtra", "Delhi"]
    print(f"\n  {'State':15s} {'Risk':8s} {'Level':10s}")
    print(f"  {'-'*35}")
    for state in states:
        state_ind = pipeline.get_state_scenario(state)
        result = predictor.predict_quick(state_ind)
        bar = "█" * int(result["risk_score"] / 10)
        print(f"  {state:15s} {result['risk_score']:5.1f}%   {result['risk_level']:10s} {bar}")

    # ─── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if all_passed:
        print("  ✅ ALL VALIDATION TESTS PASSED")
    else:
        print("  ⚠️  SOME TESTS HAD UNEXPECTED LEVELS (model may need more data)")
    print(f"{'='*70}")
    print(f"\n  Model Summary:")
    print(f"    Training samples: {len(X)}")
    print(f"    Data source: Real India historical data (2000-2024)")
    print(f"    Model type: {predictor.lgbm.model_type}")
    print(f"    Historical cases in DB: {len(predictor.matcher.cases)}")
    print(f"    India-specific interventions: {len(predictor.whatif.INTERVENTIONS)}")
    print(f"\n  Model saved to: models/lgbm/")
    print(f"\n  This model was trained on REAL India data —")
    print(f"  NOT generated by any generative AI or API.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
