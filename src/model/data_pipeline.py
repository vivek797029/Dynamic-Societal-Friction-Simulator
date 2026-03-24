"""
Data Pipeline — Prepare and process training data for the friction LLM.

HEAVY MODE: Generates massive, diverse synthetic datasets covering social,
cultural, political, crossover, and election scenarios. Includes data
augmentation (paraphrase, perspective flips, severity scaling).
"""

import json
import logging
import random
from itertools import product
from pathlib import Path

logger = logging.getLogger(__name__)


# ================================================================
# GROUPS & FACTIONS
# ================================================================

GROUPS = [
    "Traditionalists", "Progressives", "Pragmatists",
    "Isolationists", "Youth_Activists", "Working_Class",
]

FACTIONS = [
    "Left Coalition", "Center Alliance", "Right Bloc",
    "Populist Movement", "Green Alliance",
]

# ================================================================
# SOCIAL / CULTURAL SCENARIO TEMPLATES
# ================================================================

SOCIAL_TEMPLATES = [
    {
        "category": "cultural_clash",
        "template": (
            "In a diverse neighborhood, {group_a} residents have long celebrated {tradition_a} "
            "every year. Recently, {group_b} newcomers have expressed concerns about {concern}, "
            "leading to tensions in the community council meetings."
        ),
        "variables": {
            "tradition_a": [
                "a harvest festival with loud music until midnight",
                "weekly community gatherings in the main park",
                "public art installations reflecting their heritage",
                "a monthly street market selling traditional goods",
                "religious ceremonies with outdoor processions",
            ],
            "concern": [
                "noise levels disturbing families",
                "exclusive use of public spaces",
                "conflicting cultural norms and expectations",
                "perceived disrespect toward local customs",
                "safety concerns during large gatherings",
            ],
        },
    },
    {
        "category": "resource_competition",
        "template": (
            "A city council must allocate limited funding between {resource_a} demanded by "
            "{group_a} and {resource_b} prioritized by {group_b}. Both groups feel their "
            "needs are being overlooked, and protests have been organized for next week."
        ),
        "variables": {
            "resource_a": [
                "affordable housing construction",
                "expanded public transit to underserved areas",
                "new community health centers",
                "youth employment programs",
                "elder care facilities",
            ],
            "resource_b": [
                "small business development grants",
                "crumbling infrastructure repairs",
                "modernized school facilities",
                "public safety and policing",
                "environmental cleanup projects",
            ],
        },
    },
    {
        "category": "migration_tension",
        "template": (
            "A wave of {migrant_type} has settled in a region predominantly occupied by "
            "{group_a}. The {group_b} community {reaction}. Local services are "
            "{service_status}, and community leaders are struggling to find common ground."
        ),
        "variables": {
            "migrant_type": [
                "economic migrants seeking better job opportunities",
                "refugees fleeing conflict in neighboring regions",
                "rural-to-urban migrants from different cultural backgrounds",
                "tech workers from other regions driving up housing costs",
                "seasonal agricultural workers from different ethnic groups",
            ],
            "reaction": [
                "feels their cultural identity is being eroded",
                "worries about job competition and wage depression",
                "has organized welcome committees but faces internal disagreement",
                "is divided between those who welcome newcomers and those who resist",
                "sees this as both an opportunity and a threat",
            ],
            "service_status": [
                "strained beyond capacity",
                "being redirected in ways that frustrate long-term residents",
                "unable to accommodate the cultural needs of all groups",
                "becoming a flashpoint for wider political debates",
            ],
        },
    },
    {
        "category": "economic_inequality",
        "template": (
            "A new {development_type} is planned in an area where {group_a} has lived for "
            "generations. {group_b} investors are driving the project, promising {promise}, "
            "but residents fear {fear}. Property values have already {property_change}."
        ),
        "variables": {
            "development_type": [
                "luxury housing complex", "tech campus", "shopping mall",
                "mixed-use development", "industrial zone",
            ],
            "promise": [
                "hundreds of new jobs", "economic revitalization",
                "modern infrastructure", "increased tax revenue",
            ],
            "fear": [
                "displacement and gentrification",
                "loss of community character",
                "environmental degradation",
                "that benefits will bypass existing residents",
            ],
            "property_change": [
                "doubled in two years", "started pricing out families",
                "become unstable and unpredictable",
            ],
        },
    },
    {
        "category": "identity_conflict",
        "template": (
            "A debate has erupted between {group_a} and {group_b} over {issue}. "
            "{group_a} argues that {argument_a}, while {group_b} counters that {argument_b}. "
            "Social media has amplified the conflict, with {social_media_effect}."
        ),
        "variables": {
            "issue": [
                "the renaming of a historical monument",
                "the inclusion of diverse perspectives in school curricula",
                "language requirements in government services",
                "the display of cultural symbols in public buildings",
                "representation in local media and entertainment",
            ],
            "argument_a": [
                "preserving heritage is essential for social cohesion",
                "their perspective has been historically marginalized",
                "traditional values provide social stability",
                "their community's contributions are being erased",
            ],
            "argument_b": [
                "progress requires updating outdated practices",
                "inclusivity should be the priority in a diverse society",
                "economic concerns matter more than symbolic gestures",
                "all communities deserve equal recognition",
            ],
            "social_media_effect": [
                "hashtags from both sides trending nationally",
                "misinformation spreading faster than facts",
                "outside agitators amplifying the conflict",
                "viral videos taken out of context inflaming tensions",
            ],
        },
    },
    {
        "category": "generational_divide",
        "template": (
            "A sharp divide has emerged between older {group_a} members and younger "
            "{group_b} members within the same community over {issue}. The older generation "
            "{older_stance}, while younger members {younger_stance}."
        ),
        "variables": {
            "issue": [
                "the direction of community leadership",
                "attitudes toward technology and social change",
                "economic priorities and wealth distribution",
                "environmental responsibility versus economic growth",
                "the role of tradition in modern society",
            ],
            "older_stance": [
                "values stability and proven methods",
                "feels their sacrifices are being dismissed",
                "worries about losing cultural roots",
            ],
            "younger_stance": [
                "demand radical change and accountability",
                "feel locked out of economic opportunity",
                "prioritize global issues over local traditions",
            ],
        },
    },
    {
        "category": "housing_crisis",
        "template": (
            "Housing costs have {change} in the past {time_period}, disproportionately "
            "affecting {group_a}. {group_b} are {group_b_action}, leading to {outcome}."
        ),
        "variables": {
            "change": ["risen 40%", "doubled", "become unaffordable for most families"],
            "time_period": ["two years", "five years", "a single decade"],
            "group_b_action": [
                "buying up properties as investments",
                "pushing for luxury developments",
                "blocking affordable housing proposals",
                "advocating for rent controls that landlords oppose",
            ],
            "outcome": [
                "growing homelessness and community anger",
                "organized rent strikes and legal battles",
                "political candidates making housing their central issue",
                "families being forced to relocate to distant areas",
            ],
        },
    },
    {
        "category": "religious_tension",
        "template": (
            "A request by {group_a} to {religious_request} in a shared public space has "
            "been met with opposition from {group_b}, who argue that {opposition_reason}. "
            "The local government is caught between {dilemma}."
        ),
        "variables": {
            "religious_request": [
                "build a new place of worship",
                "hold weekly prayer gatherings",
                "display religious symbols during a holiday",
                "broadcast a call to prayer",
            ],
            "opposition_reason": [
                "public spaces should remain secular",
                "the noise and disruption affects their daily life",
                "it sets a precedent that favors one group over others",
                "it conflicts with the area's established character",
            ],
            "dilemma": [
                "protecting religious freedom and maintaining neutrality",
                "respecting diversity and managing community cohesion",
                "legal obligations and political pressure from both sides",
            ],
        },
    },
]

# ================================================================
# POLITICAL SCENARIO TEMPLATES
# ================================================================

POLITICAL_TEMPLATES = [
    {
        "category": "policy_disagreement",
        "template": (
            "The {faction_a} has proposed {policy}, which the {faction_b} strongly opposes. "
            "The debate has escalated with {escalation}. Public opinion polls show "
            "{poll_result}, but partisan media is framing it as {media_frame}."
        ),
        "variables": {
            "policy": [
                "a universal basic income program funded by wealth taxes",
                "strict immigration controls with deportation quotas",
                "a carbon tax with dividends returned to citizens",
                "deregulation of the technology sector",
                "nationalization of key energy companies",
                "mandatory military service for all citizens",
                "a public healthcare expansion replacing private insurance",
            ],
            "escalation": [
                "legislators walking out of parliamentary sessions",
                "street protests in multiple cities",
                "a media war between partisan outlets",
                "threats of a government shutdown",
                "leaked internal documents fueling public outrage",
            ],
            "poll_result": [
                "a deeply divided electorate with a razor-thin margin",
                "majority support but fierce minority opposition",
                "shifting opinions based on how the question is framed",
                "generational splits with young voters strongly in favor",
            ],
            "media_frame": [
                "a battle for the soul of the nation",
                "an attack on fundamental freedoms",
                "a necessary step toward modernization",
                "a power grab by political elites",
            ],
        },
    },
    {
        "category": "election_controversy",
        "template": (
            "Following a closely contested election, the {faction_a} has {action} while the "
            "{faction_b} {response}. {consequence}. Trust in democratic institutions "
            "has {trust_change}."
        ),
        "variables": {
            "action": [
                "declared victory despite contested ballot counts",
                "called for a full recount in key districts",
                "accused the opposing side of voter suppression",
                "pushed through certification despite irregularities",
            ],
            "response": [
                "is demanding international observers for a recount",
                "has filed legal challenges in multiple courts",
                "has called for mass protests outside government buildings",
                "is refusing to recognize the results",
            ],
            "consequence": [
                "Social media is flooded with conspiracy theories from both sides",
                "International organizations have expressed concern about democratic norms",
                "Business leaders are urging calm while privately choosing sides",
                "Security forces are on heightened alert in major cities",
            ],
            "trust_change": [
                "plummeted to historic lows",
                "become deeply partisan — trusted only by the winning side",
                "eroded further after years of declining confidence",
            ],
        },
    },
    {
        "category": "disinformation_campaign",
        "template": (
            "A coordinated disinformation campaign has spread {disinfo_type} targeting "
            "{faction_a} supporters. The content, originating from {origin}, has been "
            "{spread_method}. {faction_b} is {faction_b_response}."
        ),
        "variables": {
            "disinfo_type": [
                "fabricated scandal documents about party leaders",
                "doctored videos making candidates appear to say extreme things",
                "fake grassroots movements amplified by bot networks",
                "misleading statistics about economic performance",
            ],
            "origin": [
                "anonymous social media accounts with no clear attribution",
                "foreign-linked troll farms identified by researchers",
                "domestic political operatives using shell organizations",
                "AI-generated content factories producing thousands of posts daily",
            ],
            "spread_method": [
                "shared millions of times before fact-checkers could respond",
                "amplified by mainstream media outlets before verification",
                "embedded in viral entertainment content to bypass critical thinking",
                "targeted at swing voters through micro-targeted advertising",
            ],
            "faction_b_response": [
                "accused of being behind it, further muddying the waters",
                "struggling to counter the narrative despite evidence",
                "capitalizing on the chaos to push their own agenda",
                "calling for emergency regulation of social media platforms",
            ],
        },
    },
    {
        "category": "protest_movement",
        "template": (
            "A massive protest movement organized by supporters of the {faction_a} has "
            "erupted over {trigger}. The protests have {protest_status}. The {faction_b}-led "
            "government has responded with {government_response}."
        ),
        "variables": {
            "trigger": [
                "police brutality against a minority community member",
                "a controversial court ruling on civil rights",
                "proposed austerity measures cutting social programs",
                "government corruption exposed by investigative journalists",
                "environmental destruction approved by corporate-backed politicians",
            ],
            "protest_status": [
                "spread to dozens of cities with millions participating",
                "been largely peaceful but with isolated incidents of property damage",
                "created an encampment in the capital that has lasted weeks",
                "evolved into a general strike affecting key economic sectors",
            ],
            "government_response": [
                "deploying riot police and imposing curfews",
                "offering dialogue while stalling on concrete demands",
                "labeling protesters as extremists to justify a crackdown",
                "making symbolic concessions while ignoring structural demands",
            ],
        },
    },
    {
        "category": "legislative_deadlock",
        "template": (
            "A critical {legislation_type} has been blocked for {duration} due to "
            "irreconcilable differences between the {faction_a} and {faction_b}. "
            "The deadlock is causing {impact}. Public frustration is {frustration_level}."
        ),
        "variables": {
            "legislation_type": [
                "national budget", "healthcare reform bill",
                "climate action plan", "immigration reform package",
                "education funding bill", "infrastructure investment act",
            ],
            "duration": [
                "six months", "over a year", "the entire legislative session",
            ],
            "impact": [
                "government agencies to run on emergency funding",
                "essential services to deteriorate",
                "international credibility to suffer",
                "markets to become increasingly volatile",
            ],
            "frustration_level": [
                "boiling over into anti-establishment sentiment",
                "driving voters toward extremist alternatives",
                "manifesting as record-low approval ratings for all parties",
                "creating openings for populist outsiders",
            ],
        },
    },
    {
        "category": "corruption_scandal",
        "template": (
            "Senior leaders of the {faction_a} have been implicated in {scandal_type}. "
            "Evidence released by {evidence_source} shows {evidence_detail}. The {faction_b} "
            "is {response}, while the public is {public_reaction}."
        ),
        "variables": {
            "scandal_type": [
                "accepting millions in undisclosed corporate donations",
                "awarding government contracts to personal associates",
                "using classified information for private financial gain",
                "covering up environmental contamination to protect donors",
            ],
            "evidence_source": [
                "investigative journalists", "a whistleblower inside the party",
                "leaked internal communications", "a judicial inquiry",
            ],
            "evidence_detail": [
                "a systematic pattern of corruption over multiple years",
                "direct links between policy decisions and private payments",
                "attempts to obstruct investigations and silence witnesses",
            ],
            "response": [
                "demanding immediate resignations and criminal prosecution",
                "using it as a campaign weapon while ignoring their own issues",
                "calling for a bipartisan ethics investigation",
            ],
            "public_reaction": [
                "deeply cynical, believing all politicians are corrupt",
                "sharply divided along partisan lines about the severity",
                "demanding systemic reform beyond individual accountability",
            ],
        },
    },
]

# ================================================================
# CROSSOVER TEMPLATES (Social + Political)
# ================================================================

CROSSOVER_TEMPLATES = [
    {
        "category": "politicized_cultural_issue",
        "template": (
            "What began as a local dispute between {group_a} and {group_b} over {local_issue} "
            "has been seized upon by the {faction_a} and {faction_b} as a national political "
            "issue. {faction_a} frames it as {frame_a}, while {faction_b} presents it as "
            "{frame_b}. The original community members feel {community_feeling}."
        ),
        "variables": {
            "local_issue": [
                "a school board's curriculum decisions",
                "a zoning dispute about a cultural center",
                "a workplace dress code policy",
                "the language used in government communications",
                "public funding for cultural events",
            ],
            "frame_a": [
                "a fundamental rights issue that demands national legislation",
                "evidence of systemic discrimination requiring immediate action",
                "a threat to national identity that must be stopped",
            ],
            "frame_b": [
                "government overreach into local community matters",
                "a distraction from real economic issues affecting everyone",
                "a manufactured crisis designed to win votes",
            ],
            "community_feeling": [
                "their voices have been drowned out by national politics",
                "used as pawns in a game they never asked to play",
                "both empowered and overwhelmed by the sudden attention",
            ],
        },
    },
    {
        "category": "climate_policy_vs_jobs",
        "template": (
            "A proposed {climate_action} would significantly impact {group_a} workers in "
            "{industry}. The {faction_a} argues this is essential for survival, while the "
            "{faction_b} warns of {economic_warning}. {group_b} is caught in the middle, "
            "{middle_position}."
        ),
        "variables": {
            "climate_action": [
                "ban on fossil fuel extraction by 2035",
                "carbon border tax on imported goods",
                "mandatory green energy transition for all industries",
                "closure of coal plants within five years",
            ],
            "industry": [
                "the energy sector", "manufacturing", "agriculture",
                "transportation and logistics", "mining communities",
            ],
            "economic_warning": [
                "mass unemployment in already struggling regions",
                "energy prices that will devastate working families",
                "a competitive disadvantage against nations without such policies",
            ],
            "middle_position": [
                "wanting environmental protection but fearing job losses",
                "supporting transition but demanding adequate retraining programs",
                "feeling abandoned by both sides of the political spectrum",
            ],
        },
    },
    {
        "category": "immigration_policy_debate",
        "template": (
            "A surge in {migration_type} has become the dominant issue in national politics. "
            "{group_a} communities are directly affected by {direct_impact}. The {faction_a} "
            "proposes {proposal_a}, while the {faction_b} advocates {proposal_b}. "
            "Tensions have led to {tension_outcome}."
        ),
        "variables": {
            "migration_type": [
                "asylum seekers from conflict zones",
                "economic migrants from neighboring countries",
                "climate refugees from increasingly uninhabitable regions",
            ],
            "direct_impact": [
                "overcrowded local services and housing shortages",
                "changing neighborhood demographics and cultural friction",
                "competition for low-wage jobs and downward pressure on salaries",
            ],
            "proposal_a": [
                "expanded pathways to citizenship and integration programs",
                "strict border enforcement and deportation of undocumented residents",
                "a points-based immigration system prioritizing economic needs",
            ],
            "proposal_b": [
                "a humanitarian approach with open borders",
                "a total immigration moratorium until infrastructure catches up",
                "regional agreements to share the burden more equitably",
            ],
            "tension_outcome": [
                "violent incidents between communities and calls for military intervention",
                "local communities forming vigilante groups and counter-protests",
                "a nationwide debate about national identity and belonging",
            ],
        },
    },
    {
        "category": "education_curriculum_war",
        "template": (
            "A fierce battle has erupted over {curriculum_issue} in public schools. {group_a} "
            "parents demand {demand_a}, while {group_b} parents insist on {demand_b}. "
            "The {faction_a} has introduced legislation to {legislation}, drawing {reaction} "
            "from educators and civil society."
        ),
        "variables": {
            "curriculum_issue": [
                "how national history is taught, including past injustices",
                "sex education content and age-appropriateness",
                "the inclusion of diverse religious perspectives",
                "climate science education versus industry-friendly narratives",
            ],
            "demand_a": [
                "preservation of traditional educational values",
                "honest and complete inclusion of marginalized histories",
                "parental choice in what children are exposed to",
            ],
            "demand_b": [
                "modernized curricula reflecting scientific consensus",
                "protection of children from politically motivated content",
                "curricula that prepare students for a diverse workforce",
            ],
            "legislation": [
                "ban certain topics from being discussed in classrooms",
                "mandate specific content that aligns with their ideology",
                "give parents veto power over curriculum decisions",
            ],
            "reaction": [
                "mass teacher resignations and protests",
                "legal challenges from civil liberties organizations",
                "counter-legislation from the opposing party in other states",
            ],
        },
    },
]

# ================================================================
# ELECTION TEMPLATES
# ================================================================

ELECTION_TEMPLATES = [
    {
        "category": "election_campaign",
        "template": (
            "The upcoming election is the most polarized in decades. The {faction_a} candidate "
            "is running on a platform of {platform_a}, appealing to {group_a} and {group_b} "
            "voters. The {faction_b} candidate counters with {platform_b}, targeting "
            "{group_c} communities. Campaign rhetoric has {rhetoric_status}."
        ),
        "variables": {
            "platform_a": [
                "radical economic transformation and wealth redistribution",
                "restoring traditional values and national pride",
                "pragmatic centrism and cross-party cooperation",
                "anti-establishment reform and draining institutional corruption",
            ],
            "platform_b": [
                "fiscal responsibility and limited government intervention",
                "progressive social policies and environmental leadership",
                "security-first governance and border protection",
                "grassroots democracy and direct citizen participation",
            ],
            "rhetoric_status": [
                "descended into personal attacks and character assassination",
                "become increasingly apocalyptic — each side claims the other will destroy the country",
                "been amplified by social media echo chambers into extreme positions",
                "alienated centrist voters who feel neither side represents them",
            ],
        },
    },
]


# ================================================================
# ANALYSIS INSTRUCTIONS
# ================================================================

ANALYSIS_ASPECTS = [
    "friction_type", "escalation_risk", "root_causes",
    "affected_parties", "potential_resolutions", "emotional_dynamics",
    "historical_parallels", "structural_factors", "media_influence",
    "political_implications", "economic_consequences",
    "long_term_trajectory", "intervention_points",
    "stakeholder_motivations", "power_dynamics",
]


# ================================================================
# GENERATION FUNCTIONS
# ================================================================

def _fill_template(template_info: dict, domain: str) -> dict:
    """Fill a template with random variables and groups/factions."""
    text = template_info["template"]

    # Replace group placeholders
    groups = random.sample(GROUPS, min(3, len(GROUPS)))
    for i, placeholder in enumerate(["group_a", "group_b", "group_c"]):
        if f"{{{placeholder}}}" in text and i < len(groups):
            text = text.replace(f"{{{placeholder}}}", groups[i])

    # Replace faction placeholders
    factions = random.sample(FACTIONS, min(2, len(FACTIONS)))
    for i, placeholder in enumerate(["faction_a", "faction_b"]):
        if f"{{{placeholder}}}" in text and i < len(factions):
            text = text.replace(f"{{{placeholder}}}", factions[i])

    # Fill in template variables
    for var_name, options in template_info.get("variables", {}).items():
        if f"{{{var_name}}}" in text:
            text = text.replace(f"{{{var_name}}}", random.choice(options))

    return {
        "scenario": text,
        "category": template_info["category"],
        "domain": domain,
        "groups": groups[:2],
        "factions": factions[:2] if domain in ("political", "crossover", "election") else [],
        "severity": round(random.uniform(0.2, 0.95), 2),
    }


def generate_synthetic_scenario() -> dict:
    """Generate a single scenario from a randomly selected domain."""
    domain_weights = {
        "social": (SOCIAL_TEMPLATES, 0.35),
        "political": (POLITICAL_TEMPLATES, 0.35),
        "crossover": (CROSSOVER_TEMPLATES, 0.20),
        "election": (ELECTION_TEMPLATES, 0.10),
    }

    domains = list(domain_weights.keys())
    weights = [domain_weights[d][1] for d in domains]
    chosen_domain = random.choices(domains, weights=weights, k=1)[0]
    templates = domain_weights[chosen_domain][0]

    template_info = random.choice(templates)
    return _fill_template(template_info, chosen_domain)


def format_as_instruction(scenario: dict) -> dict:
    """Format a scenario into instruction-tuning format."""
    num_aspects = random.randint(4, 7)
    aspects = random.sample(ANALYSIS_ASPECTS, k=min(num_aspects, len(ANALYSIS_ASPECTS)))

    domain_label = scenario["domain"].upper()
    instruction = (
        f"[{domain_label} SCENARIO]\n\n"
        f"Analyze the following friction scenario.\n\n"
        f"Scenario: {scenario['scenario']}\n\n"
        f"Provide a detailed, multi-perspective analysis covering: {', '.join(aspects)}.\n"
        f"Consider the viewpoints of all affected groups and factions."
    )

    # Placeholder response — replace with teacher model output in production
    response_placeholder = (
        f"[ANALYSIS PLACEHOLDER — Replace with expert annotation or teacher-model output]\n"
        f"Domain: {scenario['domain']}\n"
        f"Category: {scenario['category']}\n"
        f"Groups: {', '.join(scenario['groups'])}\n"
        f"Factions: {', '.join(scenario['factions']) if scenario['factions'] else 'N/A'}\n"
        f"Severity: {scenario['severity']}\n"
        f"Required aspects: {', '.join(aspects)}"
    )

    return {
        "instruction": instruction,
        "input": "",
        "output": response_placeholder,
        "domain": scenario["domain"],
        "category": scenario["category"],
        "metadata": {
            "groups": scenario["groups"],
            "factions": scenario["factions"],
            "severity": scenario["severity"],
        },
    }


def augment_sample(sample: dict) -> list[dict]:
    """
    Generate augmented versions of a single sample.

    - Perspective flip: same scenario from the other group's point of view
    - Severity scaling: same scenario at different severity levels
    """
    augmented = []

    # Perspective flip
    groups = sample["metadata"].get("groups", [])
    if len(groups) >= 2:
        flipped = sample.copy()
        flipped["metadata"] = sample["metadata"].copy()
        flipped["metadata"]["groups"] = list(reversed(groups))
        flipped["instruction"] = sample["instruction"].replace(
            f"Scenario: ", f"Scenario (from {groups[1]} perspective): "
        )
        augmented.append(flipped)

    # Severity scaling — generate a low and high severity version
    for new_severity in [0.3, 0.85]:
        scaled = sample.copy()
        scaled["metadata"] = sample["metadata"].copy()
        scaled["metadata"]["severity"] = new_severity
        severity_label = "LOW" if new_severity < 0.5 else "HIGH"
        scaled["instruction"] = sample["instruction"].replace(
            "Analyze the following",
            f"[{severity_label} SEVERITY] Analyze the following",
        )
        augmented.append(scaled)

    return augmented


def generate_dataset(
    num_samples: int = 1000,
    output_dir: str = "data/processed",
    eval_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
):
    """Generate a synthetic training dataset with optional augmentation."""
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_samples} base samples...")

    # Generate base samples
    base_samples = []
    for i in range(num_samples):
        scenario = generate_synthetic_scenario()
        sample = format_as_instruction(scenario)
        base_samples.append(sample)

        if (i + 1) % 5000 == 0:
            logger.info(f"  Generated {i + 1}/{num_samples} base samples...")

    # Domain distribution stats
    domain_counts = {}
    for s in base_samples:
        domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1
    logger.info(f"Domain distribution: {domain_counts}")

    # Augmentation
    all_samples = list(base_samples)
    if augment:
        logger.info("Augmenting dataset (perspective flips + severity scaling)...")
        for sample in base_samples:
            augmented = augment_sample(sample)
            all_samples.extend(augmented)
        logger.info(f"Augmented: {len(base_samples)} → {len(all_samples)} total samples")

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - eval_ratio))
    train_data = all_samples[:split_idx]
    eval_data = all_samples[split_idx:]

    # Save
    train_path = out / "train.jsonl"
    eval_path = out / "eval.jsonl"

    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    logger.info(f"Saved {len(train_data)} train + {len(eval_data)} eval samples")
    logger.info(f"Files: {train_path}, {eval_path}")

    return {"train_samples": len(train_data), "eval_samples": len(eval_data)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dataset(num_samples=50000, augment=True)
