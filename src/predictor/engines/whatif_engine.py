"""
What-If Causal Analysis Engine — Layer 5
Tests government interventions BEFORE implementing them.

Input:  Current conditions + Proposed intervention
Output: New risk score, risk change, side effects

Example:
  "If government increases MSP by 20%"
  → Farmer tension: -30%, Inflation risk: +5%
  → Net risk: drops from HIGH (78%) to MEDIUM (48%)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy


class WhatIfEngine:
    """
    Counterfactual analysis engine for policy intervention testing.
    Based on causal inference principles from conflict research.
    """

    # India-specific intervention catalog with causal effects on indicators
    # Each intervention maps to real Government of India schemes and policies
    INTERVENTIONS = {
        "increase_msp": {
            "name": "Increase MSP (Minimum Support Price) by 10-20%",
            "category": "agrarian",
            "description": "Government raises MSP for kharif/rabi crops — directly benefits 10 Cr+ farmers",
            "effects": {
                "food_price_index": {"change_pct": +4, "delay_months": 1},
                "poverty_rate": {"change_pct": -10, "delay_months": 3},
                "inflation_rate": {"change_pct": +3, "delay_months": 2},
                "gdp_growth": {"change_pct": +1, "delay_months": 6}
            },
            "risk_modifier": -18,
            "side_effects": ["Consumer food inflation may rise 2-3%", "Fiscal cost ~₹15,000 Cr",
                            "Benefits mainly wheat/rice farmers, not horticulture"],
            "effectiveness": 0.78,
            "cost": "medium",
            "speed": "medium",
            "scheme": "CACP MSP Recommendation"
        },
        "expand_mgnrega": {
            "name": "MGNREGA Expansion — 100 to 150 days + wage hike",
            "category": "employment",
            "description": "Expand rural employment guarantee from 100 to 150 days, raise daily wage to ₹350",
            "effects": {
                "unemployment_rate": {"change_pct": -15, "delay_months": 2},
                "poverty_rate": {"change_pct": -12, "delay_months": 3},
                "gdp_growth": {"change_pct": +2, "delay_months": 6},
                "gini_coefficient": {"change_pct": -5, "delay_months": 6}
            },
            "risk_modifier": -22,
            "side_effects": ["Fiscal cost ~₹1.2 Lakh Cr", "May reduce farm labor supply",
                            "Implementation leakage in some states"],
            "effectiveness": 0.82,
            "cost": "high",
            "speed": "fast",
            "scheme": "MGNREGA"
        },
        "pm_kisan_boost": {
            "name": "PM-KISAN Enhancement — ₹6000 to ₹10,000/year",
            "category": "agrarian",
            "description": "Increase direct cash transfer to 9.5 Cr farmer families via DBT",
            "effects": {
                "poverty_rate": {"change_pct": -8, "delay_months": 1},
                "food_price_index": {"change_pct": -3, "delay_months": 2},
                "gdp_growth": {"change_pct": +1, "delay_months": 3}
            },
            "risk_modifier": -14,
            "side_effects": ["Fiscal cost ~₹38,000 Cr additional", "Doesn't address structural issues",
                            "Landless laborers still excluded"],
            "effectiveness": 0.72,
            "cost": "high",
            "speed": "fast",
            "scheme": "PM-KISAN"
        },
        "pmgkay_activation": {
            "name": "PMGKAY — Free Ration to 80 Cr People",
            "category": "food_security",
            "description": "Activate PM Garib Kalyan Anna Yojana — 5kg free grain per person per month",
            "effects": {
                "food_price_index": {"change_pct": -20, "delay_months": 1},
                "poverty_rate": {"change_pct": -15, "delay_months": 1},
                "inflation_rate": {"change_pct": -4, "delay_months": 2}
            },
            "risk_modifier": -25,
            "side_effects": ["Fiscal cost ~₹3.5 Lakh Cr/year", "FCI storage strain",
                            "Grain procurement drives up market prices"],
            "effectiveness": 0.88,
            "cost": "very_high",
            "speed": "fast",
            "scheme": "PMGKAY"
        },
        "skill_india_push": {
            "name": "Skill India Mission + PMKVY 4.0 Fast-Track",
            "category": "employment",
            "description": "Mass skilling drive under PMKVY — target 1 Cr youth in 6 months",
            "effects": {
                "unemployment_rate": {"change_pct": -12, "delay_months": 9},
                "youth_bulge_pct": {"change_pct": -5, "delay_months": 12},
                "poverty_rate": {"change_pct": -6, "delay_months": 12}
            },
            "risk_modifier": -10,
            "side_effects": ["Skill-job mismatch persists", "Quality of training varies",
                            "Takes 9-12 months for employment impact"],
            "effectiveness": 0.65,
            "cost": "medium",
            "speed": "slow",
            "scheme": "PMKVY 4.0"
        },
        "pli_expansion": {
            "name": "PLI Scheme Expansion — 5 New Labor-Intensive Sectors",
            "category": "industrial",
            "description": "Production Linked Incentive for textiles, toys, footwear, furniture, agro-processing",
            "effects": {
                "unemployment_rate": {"change_pct": -18, "delay_months": 12},
                "gdp_growth": {"change_pct": +10, "delay_months": 12},
                "poverty_rate": {"change_pct": -8, "delay_months": 18}
            },
            "risk_modifier": -16,
            "side_effects": ["High fiscal outlay ~₹2 Lakh Cr", "Benefits mostly organized sector",
                            "12-18 month lag before job creation"],
            "effectiveness": 0.80,
            "cost": "very_high",
            "speed": "slow",
            "scheme": "PLI Scheme"
        },
        "all_party_dialogue": {
            "name": "All-Party Meet + National Integration Council",
            "category": "political",
            "description": "PM convenes all-party meeting, activates NIC for consensus building",
            "effects": {
                "political_stability": {"change_pct": +18, "delay_months": 1},
                "press_freedom_index": {"change_pct": +8, "delay_months": 1}
            },
            "risk_modifier": -14,
            "side_effects": ["Opposition may use platform for grandstanding",
                            "Requires genuine government commitment",
                            "Effectiveness depends on PM's political will"],
            "effectiveness": 0.58,
            "cost": "low",
            "speed": "fast",
            "scheme": "Parliamentary Process"
        },
        "internet_shutdown": {
            "name": "Internet Shutdown / Section 144 Deployment",
            "category": "security",
            "description": "State government imposes internet ban + curfew in affected areas",
            "effects": {
                "political_stability": {"change_pct": +3, "delay_months": 0},
                "press_freedom_index": {"change_pct": -25, "delay_months": 0},
                "internet_penetration": {"change_pct": -50, "delay_months": 0}
            },
            "risk_modifier": -2,
            "side_effects": ["Economic loss ~₹2,000 Cr/day", "International criticism",
                            "Supreme Court has ruled against blanket shutdowns",
                            "Suppresses symptoms, amplifies long-term grievance",
                            "India leads world in internet shutdowns"],
            "effectiveness": 0.20,
            "cost": "low",
            "speed": "immediate",
            "scheme": "IT Act Section 69A / CrPC 144"
        },
        "lokpal_crackdown": {
            "name": "Lokpal/CBI Anti-Corruption Crackdown",
            "category": "governance",
            "description": "Activate Lokpal + CBI investigation into corruption allegations fueling unrest",
            "effects": {
                "corruption_index": {"change_pct": +12, "delay_months": 12},
                "political_stability": {"change_pct": +10, "delay_months": 6}
            },
            "risk_modifier": -10,
            "side_effects": ["Political class resistance", "CBI credibility concerns (caged parrot)",
                            "Long investigation timelines", "May trigger counter-accusations"],
            "effectiveness": 0.50,
            "cost": "medium",
            "speed": "slow",
            "scheme": "Lokpal & Lokayuktas Act 2013"
        },
        "reservation_reform": {
            "name": "Reservation/Quota Restructuring",
            "category": "social_equity",
            "description": "Revise OBC/EWS reservation, add new communities to scheduled list",
            "effects": {
                "political_stability": {"change_pct": +12, "delay_months": 6},
                "gini_coefficient": {"change_pct": -8, "delay_months": 12},
                "youth_bulge_pct": {"change_pct": -3, "delay_months": 12}
            },
            "risk_modifier": -12,
            "side_effects": ["Backlash from non-beneficiary communities",
                            "Legal challenges — SC 50% ceiling",
                            "Political exploitation of caste dynamics",
                            "May trigger new demands from other groups"],
            "effectiveness": 0.55,
            "cost": "low",
            "speed": "medium",
            "scheme": "Article 15(4), 16(4) Constitutional Provision"
        },
        "mudra_msme_package": {
            "name": "MUDRA + MSME Emergency Credit Package",
            "category": "economic",
            "description": "₹50,000 Cr emergency MUDRA loans + ECLGS for MSMEs in distressed areas",
            "effects": {
                "unemployment_rate": {"change_pct": -10, "delay_months": 3},
                "gdp_growth": {"change_pct": +5, "delay_months": 6},
                "poverty_rate": {"change_pct": -6, "delay_months": 6}
            },
            "risk_modifier": -14,
            "side_effects": ["NPA risk if borrowers default", "Benefits urban MSMEs more",
                            "Disbursement delays in backward districts"],
            "effectiveness": 0.72,
            "cost": "high",
            "speed": "medium",
            "scheme": "PM MUDRA Yojana + ECLGS"
        },
        "paramilitary_deployment": {
            "name": "CRPF/BSF/RAF Paramilitary Deployment",
            "category": "security",
            "description": "Central Armed Police Forces deployment to maintain law and order",
            "effects": {
                "political_stability": {"change_pct": +5, "delay_months": 0},
                "press_freedom_index": {"change_pct": -12, "delay_months": 0}
            },
            "risk_modifier": -4,
            "side_effects": ["May escalate tensions if force used disproportionately",
                            "NHRC/SHRC complaints likely", "International media scrutiny",
                            "Addresses symptoms not root causes",
                            "Historical precedent: Jallianwala Bagh effect"],
            "effectiveness": 0.30,
            "cost": "medium",
            "speed": "immediate",
            "scheme": "Article 355/356, CRPF Act"
        },
        "ayushman_expansion": {
            "name": "Ayushman Bharat PM-JAY Expansion to Affected Areas",
            "category": "social_welfare",
            "description": "Fast-track PM-JAY health insurance for affected communities — ₹5 Lakh cover",
            "effects": {
                "poverty_rate": {"change_pct": -5, "delay_months": 2},
                "political_stability": {"change_pct": +5, "delay_months": 3}
            },
            "risk_modifier": -8,
            "side_effects": ["Hospital infrastructure may be insufficient",
                            "Empaneled hospital coverage varies by state",
                            "Doesn't address immediate grievances"],
            "effectiveness": 0.60,
            "cost": "medium",
            "speed": "medium",
            "scheme": "PM-JAY (Ayushman Bharat)"
        }
    }

    def analyze_intervention(self, indicators: Dict, intervention_id: str,
                             current_risk: float, magnitude: float = 1.0) -> Dict:
        """Analyze the impact of a specific intervention.

        Args:
            indicators: Current socioeconomic indicators
            intervention_id: Key from INTERVENTIONS catalog
            current_risk: Current risk score (0-100)
            magnitude: Scale factor for intervention (0.5 = half, 2.0 = double)

        Returns:
            Dict with before/after comparison, risk change, side effects
        """
        if intervention_id not in self.INTERVENTIONS:
            return {"error": f"Unknown intervention: {intervention_id}",
                    "available": list(self.INTERVENTIONS.keys())}

        intervention = self.INTERVENTIONS[intervention_id]
        new_indicators = deepcopy(indicators)

        # Apply effects to indicators
        indicator_changes = []
        for indicator_name, effect in intervention["effects"].items():
            if indicator_name in new_indicators:
                old_val = float(new_indicators[indicator_name])
                change = old_val * (effect["change_pct"] / 100) * magnitude
                new_val = old_val + change

                # Clamp
                if indicator_name in ["political_stability", "press_freedom_index", "gini_coefficient"]:
                    new_val = np.clip(new_val, 0, 1)
                elif indicator_name in ["corruption_index"]:
                    new_val = np.clip(new_val, 0, 100)
                else:
                    new_val = max(0, new_val)

                new_indicators[indicator_name] = round(float(new_val), 2)

                indicator_changes.append({
                    "indicator": indicator_name,
                    "before": round(old_val, 2),
                    "after": round(float(new_val), 2),
                    "change": round(float(change), 2),
                    "change_pct": round(effect["change_pct"] * magnitude, 1),
                    "delay_months": effect["delay_months"],
                    "direction": "improved" if self._is_improvement(indicator_name, change) else "worsened"
                })

        # Calculate new risk
        risk_change = intervention["risk_modifier"] * magnitude * intervention["effectiveness"]
        new_risk = np.clip(current_risk + risk_change, 0, 100)

        # Risk level labels
        old_level = self._risk_level(current_risk)
        new_level = self._risk_level(new_risk)

        return {
            "intervention": {
                "id": intervention_id,
                "name": intervention["name"],
                "category": intervention["category"],
                "magnitude": magnitude,
                "effectiveness": intervention["effectiveness"],
                "cost": intervention["cost"],
                "speed": intervention["speed"]
            },
            "risk_before": round(float(current_risk), 1),
            "risk_after": round(float(new_risk), 1),
            "risk_change": round(float(new_risk - current_risk), 1),
            "risk_change_pct": round(float((new_risk - current_risk) / max(current_risk, 1) * 100), 1),
            "level_before": old_level,
            "level_after": new_level,
            "level_improved": self._severity_rank(new_level) < self._severity_rank(old_level),
            "indicator_changes": indicator_changes,
            "new_indicators": new_indicators,
            "side_effects": intervention["side_effects"]
        }

    def analyze_multiple_interventions(self, indicators: Dict, intervention_ids: List[str],
                                        current_risk: float) -> Dict:
        """Analyze combined effect of multiple interventions."""
        cumulative_indicators = deepcopy(indicators)
        cumulative_risk = current_risk
        results = []

        for int_id in intervention_ids:
            result = self.analyze_intervention(cumulative_indicators, int_id, cumulative_risk)
            if "error" not in result:
                cumulative_indicators = result["new_indicators"]
                cumulative_risk = result["risk_after"]
                results.append(result)

        return {
            "interventions_applied": len(results),
            "original_risk": round(float(current_risk), 1),
            "final_risk": round(float(cumulative_risk), 1),
            "total_risk_change": round(float(cumulative_risk - current_risk), 1),
            "original_level": self._risk_level(current_risk),
            "final_level": self._risk_level(cumulative_risk),
            "individual_results": results,
            "final_indicators": cumulative_indicators
        }

    def recommend_interventions(self, indicators: Dict, current_risk: float,
                                 budget: str = "medium") -> List[Dict]:
        """Recommend best interventions based on current situation.

        Args:
            indicators: Current indicators
            current_risk: Current risk score
            budget: "low", "medium", "high"

        Returns:
            Sorted list of recommended interventions
        """
        budget_filter = {
            "low": ["low"],
            "medium": ["low", "medium"],
            "high": ["low", "medium", "high"]
        }
        allowed_costs = budget_filter.get(budget, ["low", "medium", "high"])

        recommendations = []
        for int_id, intervention in self.INTERVENTIONS.items():
            if intervention["cost"] not in allowed_costs:
                continue

            result = self.analyze_intervention(indicators, int_id, current_risk)
            if "error" in result:
                continue

            # Score = risk reduction * effectiveness / cost factor
            cost_factor = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
            score = abs(result["risk_change"]) * intervention["effectiveness"]
            score /= cost_factor.get(intervention["cost"], 2)

            recommendations.append({
                "intervention_id": int_id,
                "name": intervention["name"],
                "score": round(float(score), 2),
                "risk_reduction": round(abs(result["risk_change"]), 1),
                "effectiveness": intervention["effectiveness"],
                "cost": intervention["cost"],
                "speed": intervention["speed"],
                "side_effects_count": len(intervention["side_effects"])
            })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations

    def get_available_interventions(self) -> List[Dict]:
        """Get all available interventions."""
        return [
            {"id": k, "name": v["name"], "category": v["category"],
             "cost": v["cost"], "speed": v["speed"], "effectiveness": v["effectiveness"]}
            for k, v in self.INTERVENTIONS.items()
        ]

    def _is_improvement(self, indicator: str, change: float) -> bool:
        """Check if a change is an improvement."""
        worse_when_higher = ["unemployment_rate", "inflation_rate", "gini_coefficient",
                            "youth_bulge_pct", "poverty_rate", "food_price_index",
                            "military_expenditure_pct"]
        if indicator in worse_when_higher:
            return change < 0
        return change > 0

    def _risk_level(self, score: float) -> str:
        if score >= 80: return "CRITICAL"
        elif score >= 60: return "HIGH"
        elif score >= 40: return "MEDIUM"
        else: return "LOW"

    def _severity_rank(self, level: str) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(level, 1)
