"""
Policy Recommendation Engine
Generates actionable policy recommendations based on risk analysis.

Timeline: Immediate (Week 1) → Short-term (Month 1) → Medium-term (3 months)
Each recommendation includes cost, impact, and priority level.
"""

import numpy as np
from typing import Dict, List
from ..data.historical_cases import INDICATOR_METADATA


class PolicyEngine:
    """
    Generates structured policy recommendations based on:
    1. Current risk factors
    2. Historical case resolutions
    3. Indicator severity levels
    4. Budget constraints
    """

    # India-specific policy database organized by friction type and timeline
    # References real Government of India schemes and institutional mechanisms
    POLICY_DATABASE = {
        "economic_crisis": {
            "immediate": [
                {"action": "Activate PMGKAY (PM Garib Kalyan Anna Yojana) — free ration to 80 Cr beneficiaries",
                 "cost": "high", "impact": "high", "target_indicator": "food_price_index",
                 "scheme": "PMGKAY", "ministry": "Ministry of Consumer Affairs"},
                {"action": "RBI emergency repo rate intervention + CRR adjustment",
                 "cost": "low", "impact": "medium", "target_indicator": "inflation_rate",
                 "scheme": "RBI Monetary Policy", "ministry": "Reserve Bank of India"},
                {"action": "Release buffer stock of wheat & rice through FCI at subsidized rates",
                 "cost": "medium", "impact": "high", "target_indicator": "food_price_index",
                 "scheme": "FCI Buffer Stock", "ministry": "Ministry of Food & Public Distribution"}
            ],
            "short_term": [
                {"action": "Increase PM-KISAN installment from ₹6000 to ₹8000/year for farmer relief",
                 "cost": "high", "impact": "high", "target_indicator": "poverty_rate",
                 "scheme": "PM-KISAN", "ministry": "Ministry of Agriculture"},
                {"action": "Expand MGNREGA guarantee from 100 to 150 days in distressed districts",
                 "cost": "high", "impact": "high", "target_indicator": "unemployment_rate",
                 "scheme": "MGNREGA", "ministry": "Ministry of Rural Development"},
                {"action": "Emergency MUDRA loan disbursement for MSMEs — ₹50,000 Cr package",
                 "cost": "high", "impact": "high", "target_indicator": "gdp_growth",
                 "scheme": "PM MUDRA Yojana", "ministry": "Ministry of Finance"}
            ],
            "medium_term": [
                {"action": "Fast-track PLI (Production Linked Incentive) in labor-intensive sectors",
                 "cost": "very_high", "impact": "very_high", "target_indicator": "gdp_growth",
                 "scheme": "PLI Scheme", "ministry": "DPIIT"},
                {"action": "Expand Make in India corridors — new industrial clusters in backward districts",
                 "cost": "very_high", "impact": "very_high", "target_indicator": "gdp_growth",
                 "scheme": "Make in India", "ministry": "DPIIT"},
                {"action": "GST rate rationalization — reduce rates on essential commodities",
                 "cost": "medium", "impact": "high", "target_indicator": "gini_coefficient",
                 "scheme": "GST Council", "ministry": "Ministry of Finance"}
            ]
        },
        "unemployment_crisis": {
            "immediate": [
                {"action": "Emergency MGNREGA wage increase + instant enrollment in rural areas",
                 "cost": "high", "impact": "high", "target_indicator": "unemployment_rate",
                 "scheme": "MGNREGA", "ministry": "Ministry of Rural Development"},
                {"action": "Activate Skill India Mission — fast-track PMKVY 4.0 certification drives",
                 "cost": "medium", "impact": "medium", "target_indicator": "unemployment_rate",
                 "scheme": "PMKVY", "ministry": "Ministry of Skill Development"},
                {"action": "Announce Rozgar Mela — 10 Lakh government recruitment drive",
                 "cost": "high", "impact": "high", "target_indicator": "unemployment_rate",
                 "scheme": "Rozgar Mela", "ministry": "DoPT"}
            ],
            "short_term": [
                {"action": "Launch PM Vishwakarma + MUDRA combined package for artisan/MSME employment",
                 "cost": "medium", "impact": "high", "target_indicator": "unemployment_rate",
                 "scheme": "PM Vishwakarma + MUDRA", "ministry": "Ministry of MSME"},
                {"action": "Expand Startup India — incubation hubs in Tier-2/Tier-3 cities",
                 "cost": "medium", "impact": "medium", "target_indicator": "youth_bulge_pct",
                 "scheme": "Startup India", "ministry": "DPIIT"},
                {"action": "PLI expansion to 5 new labor-intensive sectors (textiles, toys, footwear)",
                 "cost": "high", "impact": "high", "target_indicator": "unemployment_rate",
                 "scheme": "PLI Scheme", "ministry": "DPIIT"}
            ],
            "medium_term": [
                {"action": "National Industrial Corridor (DMIC, CBIC) — fast-track SEZ activation",
                 "cost": "very_high", "impact": "very_high", "target_indicator": "unemployment_rate",
                 "scheme": "Industrial Corridors", "ministry": "NICDC"},
                {"action": "NEP 2020 implementation — vocational education from Class 6, industry tie-ups",
                 "cost": "medium", "impact": "high", "target_indicator": "youth_bulge_pct",
                 "scheme": "NEP 2020", "ministry": "Ministry of Education"},
                {"action": "Semiconductor fab + electronics manufacturing push under DLI scheme",
                 "cost": "very_high", "impact": "very_high", "target_indicator": "gdp_growth",
                 "scheme": "India Semiconductor Mission", "ministry": "MeitY"}
            ]
        },
        "political_instability": {
            "immediate": [
                {"action": "All-party meeting convened by Speaker/PM — consensus building",
                 "cost": "low", "impact": "medium", "target_indicator": "political_stability",
                 "scheme": "Parliamentary Process", "ministry": "PMO"},
                {"action": "National Integration Council emergency session",
                 "cost": "low", "impact": "medium", "target_indicator": "political_stability",
                 "scheme": "NIC", "ministry": "Ministry of Home Affairs"},
                {"action": "Press Information Bureau transparency briefings — daily updates",
                 "cost": "low", "impact": "medium", "target_indicator": "press_freedom_index",
                 "scheme": "PIB", "ministry": "Ministry of I&B"}
            ],
            "short_term": [
                {"action": "Activate Scheduled Tribe/Caste welfare commissions for affected communities",
                 "cost": "low", "impact": "high", "target_indicator": "political_stability",
                 "scheme": "NCSC/NCST", "ministry": "Ministry of Social Justice"},
                {"action": "CBI/Lokpal investigation into corruption allegations",
                 "cost": "medium", "impact": "high", "target_indicator": "corruption_index",
                 "scheme": "Lokpal/CBI", "ministry": "DoPT"},
                {"action": "Election Commission — announce free and fair election timeline",
                 "cost": "low", "impact": "high", "target_indicator": "political_stability",
                 "scheme": "ECI", "ministry": "Election Commission of India"}
            ],
            "medium_term": [
                {"action": "Strengthen Panchayati Raj — devolve more funds to 73rd/74th Amendment bodies",
                 "cost": "medium", "impact": "very_high", "target_indicator": "political_stability",
                 "scheme": "Panchayati Raj", "ministry": "Ministry of Panchayati Raj"},
                {"action": "Women's Reservation Act implementation — fast-track delimitation",
                 "cost": "low", "impact": "high", "target_indicator": "political_stability",
                 "scheme": "Nari Shakti Vandan Act", "ministry": "Ministry of Law & Justice"},
                {"action": "Judicial reforms — increase High Court/Supreme Court bench strength",
                 "cost": "medium", "impact": "very_high", "target_indicator": "corruption_index",
                 "scheme": "Judicial Reform", "ministry": "Ministry of Law & Justice"}
            ]
        },
        "social_tension": {
            "immediate": [
                {"action": "Deploy District Magistrate-level peace committees under CrPC Section 163",
                 "cost": "low", "impact": "medium", "target_indicator": "political_stability",
                 "scheme": "District Administration", "ministry": "Ministry of Home Affairs"},
                {"action": "Activate Aman Committees + Mohalla peace groups in sensitive areas",
                 "cost": "low", "impact": "medium", "target_indicator": "political_stability",
                 "scheme": "Community Policing", "ministry": "State Police"},
                {"action": "PIB Fact Check Unit + MyGov rapid response against misinformation",
                 "cost": "low", "impact": "medium", "target_indicator": "press_freedom_index",
                 "scheme": "PIB Fact Check", "ministry": "Ministry of I&B"}
            ],
            "short_term": [
                {"action": "Expand PM-JAY (Ayushman Bharat) to cover affected community health needs",
                 "cost": "medium", "impact": "high", "target_indicator": "poverty_rate",
                 "scheme": "PM-JAY", "ministry": "Ministry of Health"},
                {"action": "Fast-track SC/ST/OBC scholarship disbursement under post-matric scheme",
                 "cost": "medium", "impact": "high", "target_indicator": "gini_coefficient",
                 "scheme": "Post-Matric Scholarship", "ministry": "Ministry of Social Justice"},
                {"action": "Launch targeted Yuva Sangam/Ek Bharat Shreshtha Bharat exchange programs",
                 "cost": "low", "impact": "medium", "target_indicator": "youth_bulge_pct",
                 "scheme": "EBSB", "ministry": "Ministry of Education"}
            ],
            "medium_term": [
                {"action": "NEP 2020 — Indian Knowledge System + multilingual education for social cohesion",
                 "cost": "medium", "impact": "high", "target_indicator": "political_stability",
                 "scheme": "NEP 2020", "ministry": "Ministry of Education"},
                {"action": "Samagra Shiksha + PM SHRI Schools in backward/conflict-prone blocks",
                 "cost": "high", "impact": "very_high", "target_indicator": "gini_coefficient",
                 "scheme": "Samagra Shiksha", "ministry": "Ministry of Education"},
                {"action": "DAY-NRLM (Deen Dayal Upadhyaya) — SHG expansion in affected communities",
                 "cost": "high", "impact": "high", "target_indicator": "poverty_rate",
                 "scheme": "DAY-NRLM", "ministry": "Ministry of Rural Development"}
            ]
        },
        "agrarian_distress": {
            "immediate": [
                {"action": "Immediate MSP procurement drive through NAFED/FCI at state mandis",
                 "cost": "high", "impact": "high", "target_indicator": "food_price_index",
                 "scheme": "MSP Procurement", "ministry": "Ministry of Agriculture"},
                {"action": "PM Fasal Bima Yojana — fast-track crop insurance claims settlement",
                 "cost": "medium", "impact": "high", "target_indicator": "poverty_rate",
                 "scheme": "PMFBY", "ministry": "Ministry of Agriculture"},
                {"action": "Emergency PM-KISAN advance installment release",
                 "cost": "medium", "impact": "medium", "target_indicator": "poverty_rate",
                 "scheme": "PM-KISAN", "ministry": "Ministry of Agriculture"}
            ],
            "short_term": [
                {"action": "Expand eNAM (National Agriculture Market) — direct farmer-to-buyer portal",
                 "cost": "medium", "impact": "high", "target_indicator": "food_price_index",
                 "scheme": "eNAM", "ministry": "Ministry of Agriculture"},
                {"action": "Kisan Credit Card limit enhancement + interest subvention",
                 "cost": "medium", "impact": "high", "target_indicator": "poverty_rate",
                 "scheme": "KCC", "ministry": "Ministry of Finance"},
                {"action": "Micro-irrigation expansion under PMKSY — Per Drop More Crop",
                 "cost": "high", "impact": "high", "target_indicator": "gdp_growth",
                 "scheme": "PMKSY", "ministry": "Ministry of Jal Shakti"}
            ],
            "medium_term": [
                {"action": "Agri-infrastructure fund — cold chain, warehousing in every block",
                 "cost": "very_high", "impact": "very_high", "target_indicator": "food_price_index",
                 "scheme": "AIF", "ministry": "Ministry of Agriculture"},
                {"action": "Farmer Producer Organization (FPO) expansion — 10,000 new FPOs",
                 "cost": "high", "impact": "very_high", "target_indicator": "gini_coefficient",
                 "scheme": "FPO Scheme", "ministry": "Ministry of Agriculture"},
                {"action": "Natural farming push under Paramparagat Krishi Vikas Yojana",
                 "cost": "medium", "impact": "high", "target_indicator": "gdp_growth",
                 "scheme": "PKVY", "ministry": "Ministry of Agriculture"}
            ]
        }
    }

    def generate_recommendations(self, indicators: Dict, risk_score: float,
                                  risk_factors: List[Dict],
                                  historical_matches: List[Dict] = None) -> Dict:
        """Generate comprehensive policy recommendations.

        Args:
            indicators: Current socioeconomic indicators
            risk_score: Current risk score
            risk_factors: Top risk factors from LGBM
            historical_matches: Matched historical cases

        Returns:
            Structured recommendations by timeline
        """
        # Identify dominant friction type
        friction_type = self._identify_friction_type(indicators, risk_factors)

        # Get base policies
        policies = self.POLICY_DATABASE.get(friction_type, self.POLICY_DATABASE["economic_crisis"])

        # Prioritize based on current indicators
        immediate = self._prioritize_actions(policies["immediate"], indicators, "immediate")
        short_term = self._prioritize_actions(policies["short_term"], indicators, "short_term")
        medium_term = self._prioritize_actions(policies["medium_term"], indicators, "medium_term")

        # Add historical lessons
        historical_lessons = []
        if historical_matches:
            for match in historical_matches[:2]:
                historical_lessons.append({
                    "case": match.get("name", "Unknown"),
                    "resolution": match.get("resolution", "unknown"),
                    "outcome": match.get("outcome", ""),
                    "lesson": f"In {match.get('name', 'a similar case')}, "
                              f"the resolution was {match.get('resolution', 'unknown')}. "
                              f"Outcome: {match.get('outcome', 'N/A')}"
                })

        return {
            "friction_type": friction_type,
            "risk_level": self._risk_level(risk_score),
            "urgency": "CRITICAL" if risk_score >= 75 else "HIGH" if risk_score >= 55 else "MODERATE",
            "recommendations": {
                "immediate": {
                    "timeline": "Within 1 week",
                    "actions": immediate
                },
                "short_term": {
                    "timeline": "Within 1 month",
                    "actions": short_term
                },
                "medium_term": {
                    "timeline": "Within 3 months",
                    "actions": medium_term
                }
            },
            "historical_lessons": historical_lessons,
            "estimated_risk_reduction": self._estimate_total_reduction(immediate + short_term + medium_term)
        }

    def _identify_friction_type(self, indicators: Dict, risk_factors: List[Dict]) -> str:
        """Identify the dominant type of friction (India-specific thresholds)."""
        scores = {
            "economic_crisis": 0,
            "unemployment_crisis": 0,
            "political_instability": 0,
            "social_tension": 0,
            "agrarian_distress": 0
        }

        # Economic indicators (tuned for Indian economy)
        if indicators.get("inflation_rate", 0) > 6:  # RBI tolerance band upper limit
            scores["economic_crisis"] += 2
        if indicators.get("food_price_index", 100) > 120:
            scores["economic_crisis"] += 2
        if indicators.get("gdp_growth", 0) < 2:  # India needs >5% to absorb labor
            scores["economic_crisis"] += 2

        # Unemployment (India has structural youth unemployment issue)
        if indicators.get("unemployment_rate", 0) > 7:  # CMIE threshold for India
            scores["unemployment_crisis"] += 3
        if indicators.get("youth_bulge_pct", 0) > 45:  # India's demographic pressure
            scores["unemployment_crisis"] += 2

        # Political
        if indicators.get("political_stability", 1) < 0.40:
            scores["political_instability"] += 3
        if indicators.get("corruption_index", 100) < 40:
            scores["political_instability"] += 2
        if indicators.get("press_freedom_index", 1) < 0.40:
            scores["political_instability"] += 1

        # Social tension (caste, communal, regional)
        if indicators.get("gini_coefficient", 0) > 0.35:  # India's inequality threshold
            scores["social_tension"] += 2
        if indicators.get("poverty_rate", 0) > 20:
            scores["social_tension"] += 2
        if indicators.get("ethnic_fractionalization", 0) > 0.40:
            scores["social_tension"] += 1

        # Agrarian distress (India-specific — 60% rural population)
        if indicators.get("food_price_index", 100) > 130:
            scores["agrarian_distress"] += 3
        if indicators.get("poverty_rate", 0) > 25 and indicators.get("urbanization_rate", 50) < 40:
            scores["agrarian_distress"] += 2
        if indicators.get("gdp_growth", 5) < 3 and indicators.get("food_price_index", 100) > 115:
            scores["agrarian_distress"] += 2

        return max(scores, key=scores.get)

    def _prioritize_actions(self, actions: List[Dict], indicators: Dict,
                            timeline: str) -> List[Dict]:
        """Prioritize and enrich action items."""
        enriched = []
        for i, action in enumerate(actions):
            target = action.get("target_indicator", "")
            current_val = indicators.get(target, 0)
            meta = INDICATOR_METADATA.get(target, {})
            threshold = meta.get("critical_threshold")

            priority = "high" if i == 0 else "medium" if i == 1 else "standard"

            enriched.append({
                "action": action["action"],
                "priority": priority,
                "cost": action["cost"],
                "expected_impact": action["impact"],
                "target_indicator": target,
                "current_value": current_val,
                "critical_threshold": threshold,
                "timeline": timeline
            })

        return enriched

    def _estimate_total_reduction(self, all_actions: List[Dict]) -> Dict:
        """Estimate total risk reduction if all recommendations implemented."""
        impact_map = {"low": 2, "medium": 5, "high": 8, "very_high": 12}
        total = sum(impact_map.get(a.get("expected_impact", "low"), 2) for a in all_actions)
        return {
            "estimated_reduction_points": min(total, 45),
            "estimated_reduction_pct": f"{min(total, 45)}%",
            "note": "Assumes full implementation of all recommendations"
        }

    def _risk_level(self, score: float) -> str:
        if score >= 80: return "CRITICAL"
        elif score >= 60: return "HIGH"
        elif score >= 40: return "MEDIUM"
        else: return "LOW"
