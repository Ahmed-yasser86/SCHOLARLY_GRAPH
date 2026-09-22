"""Evaluation question set and metric computations."""

from __future__ import annotations

QUESTION_TYPES = ("mechanism", "geographic", "temporal", "contested")

EVALUATION_QUESTIONS: list = [
    {
        "id": "mechanism-01",
        "type": "mechanism",
        "question": "How does educational inequality mediate economic inequality and intergenerational mobility?",
        "expected_mechanisms": ["educational inequality"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-02",
        "type": "mechanism",
        "question": "What role does residential segregation play between inequality and mobility?",
        "expected_mechanisms": ["residential segregation"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-03",
        "type": "mechanism",
        "question": "How does social capital transmit advantage across generations?",
        "expected_mechanisms": ["social capital"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-04",
        "type": "mechanism",
        "question": "Does wealth concentration reduce intergenerational mobility?",
        "expected_mechanisms": ["wealth concentration"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-05",
        "type": "mechanism",
        "question": "How does labor market segmentation limit upward mobility?",
        "expected_mechanisms": ["labor market segmentation"],
        "expected_countries": ["Germany"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-06",
        "type": "mechanism",
        "question": "Do political institutions mediate the Great Gatsby Curve?",
        "expected_mechanisms": ["political institutions"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-07",
        "type": "mechanism",
        "question": "How does school funding inequality affect child earnings?",
        "expected_mechanisms": ["school funding"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-08",
        "type": "mechanism",
        "question": "Does neighborhood quality change adult earnings?",
        "expected_mechanisms": ["neighborhood effects"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-09",
        "type": "mechanism",
        "question": "How do professional networks shape hiring for low-income youth?",
        "expected_mechanisms": ["professional networks"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-10",
        "type": "mechanism",
        "question": "Does inheritance explain cross-country mobility differences?",
        "expected_mechanisms": ["inheritance"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-11",
        "type": "mechanism",
        "question": "How does early childhood investment affect mobility?",
        "expected_mechanisms": ["early childhood"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-12",
        "type": "mechanism",
        "question": "Does private schooling widen mobility gaps?",
        "expected_mechanisms": ["private schooling"],
        "expected_countries": ["United Kingdom"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-13",
        "type": "mechanism",
        "question": "How do property taxes transmit inequality into school quality?",
        "expected_mechanisms": ["property taxes"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-14",
        "type": "mechanism",
        "question": "Does college completion mediate parental income effects?",
        "expected_mechanisms": ["educational attainment"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-15",
        "type": "mechanism",
        "question": "How does credential licensing segment labor markets?",
        "expected_mechanisms": ["credential barriers"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-16",
        "type": "mechanism",
        "question": "Does redistribution weaken the inequality-mobility link?",
        "expected_mechanisms": ["redistribution"],
        "expected_countries": ["Denmark"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-17",
        "type": "mechanism",
        "question": "How does healthcare access affect intergenerational outcomes?",
        "expected_mechanisms": ["healthcare access"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-18",
        "type": "mechanism",
        "question": "Does tutoring widen educational inequality?",
        "expected_mechanisms": ["tutoring"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-19",
        "type": "mechanism",
        "question": "How does vocational training shape German mobility patterns?",
        "expected_mechanisms": ["vocational training"],
        "expected_countries": ["Germany"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "mechanism-20",
        "type": "mechanism",
        "question": "Does minimum wage policy affect mobility?",
        "expected_mechanisms": ["minimum wage"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-01",
        "type": "geographic",
        "question": "How does the education mechanism differ between the United States and Sweden?",
        "expected_mechanisms": ["educational inequality"],
        "expected_countries": ["United States", "Sweden"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-02",
        "type": "geographic",
        "question": "Compare neighborhood effects in the United States and Denmark.",
        "expected_mechanisms": ["neighborhood effects"],
        "expected_countries": ["United States", "Denmark"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-03",
        "type": "geographic",
        "question": "Is wealth concentration stronger in Brazil or the United States?",
        "expected_mechanisms": ["wealth concentration"],
        "expected_countries": ["Brazil", "United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-04",
        "type": "geographic",
        "question": "How do Nordic public schools change mobility relative to the United States?",
        "expected_mechanisms": ["public education"],
        "expected_countries": ["Sweden", "United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-05",
        "type": "geographic",
        "question": "Compare informal labor markets in Brazil and Egypt.",
        "expected_mechanisms": ["informal labor markets"],
        "expected_countries": ["Brazil", "Egypt"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-06",
        "type": "geographic",
        "question": "Does social capital work differently in Germany and France?",
        "expected_mechanisms": ["social capital"],
        "expected_countries": ["Germany", "France"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-07",
        "type": "geographic",
        "question": "How does school funding compare across the United States and Finland?",
        "expected_mechanisms": ["school funding"],
        "expected_countries": ["United States", "Finland"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-08",
        "type": "geographic",
        "question": "Are tutoring effects larger in Egypt or Norway?",
        "expected_mechanisms": ["tutoring"],
        "expected_countries": ["Egypt", "Norway"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-09",
        "type": "geographic",
        "question": "Compare inheritance regimes in France and the United States.",
        "expected_mechanisms": ["inheritance"],
        "expected_countries": ["France", "United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-10",
        "type": "geographic",
        "question": "How does redistribution differ between Denmark and Brazil?",
        "expected_mechanisms": ["redistribution"],
        "expected_countries": ["Denmark", "Brazil"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-11",
        "type": "geographic",
        "question": "Do neighborhood effects replicate outside the United States?",
        "expected_mechanisms": ["neighborhood effects"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-12",
        "type": "geographic",
        "question": "Is the Great Gatsby Curve visible in developing economies?",
        "expected_mechanisms": ["great gatsby curve"],
        "expected_countries": ["Brazil"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-13",
        "type": "geographic",
        "question": "Compare vocational training in Germany and Egypt.",
        "expected_mechanisms": ["vocational training"],
        "expected_countries": ["Germany", "Egypt"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-14",
        "type": "geographic",
        "question": "How does property-tax school funding differ internationally?",
        "expected_mechanisms": ["property taxes"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "geographic-15",
        "type": "geographic",
        "question": "Does early childhood policy explain Nordic mobility?",
        "expected_mechanisms": ["early childhood"],
        "expected_countries": ["Sweden", "Denmark", "Norway"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-01",
        "type": "temporal",
        "question": "How has the inequality-mobility literature evolved since 2000?",
        "expected_mechanisms": ["economic inequality"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-02",
        "type": "temporal",
        "question": "Did mechanism research shift after Piketty's Capital?",
        "expected_mechanisms": ["wealth concentration"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-03",
        "type": "temporal",
        "question": "How have Nordic mobility estimates changed over time?",
        "expected_mechanisms": ["social mobility"],
        "expected_countries": ["Sweden"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-04",
        "type": "temporal",
        "question": "Has United States educational inequality grown since the 1990s?",
        "expected_mechanisms": ["educational inequality"],
        "expected_countries": ["United States"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-05",
        "type": "temporal",
        "question": "How did administrative tax data change mobility research?",
        "expected_mechanisms": ["earnings elasticity"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-06",
        "type": "temporal",
        "question": "Have neighborhood-effect estimates changed with new methods?",
        "expected_mechanisms": ["neighborhood effects"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-07",
        "type": "temporal",
        "question": "How has social-capital measurement evolved?",
        "expected_mechanisms": ["social capital"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-08",
        "type": "temporal",
        "question": "Did quasi-experimental designs change mechanism conclusions?",
        "expected_mechanisms": [],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-09",
        "type": "temporal",
        "question": "How has developing-country evidence grown over time?",
        "expected_mechanisms": [],
        "expected_countries": ["Brazil"],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "temporal-10",
        "type": "temporal",
        "question": "What do recent studies add to the Great Gatsby Curve?",
        "expected_mechanisms": ["great gatsby curve"],
        "expected_countries": [],
        "expect_contradiction": False,
        "relevant_papers": [],
    },
    {
        "id": "contested-01",
        "type": "contested",
        "question": "Where does the literature disagree about school funding effects?",
        "expected_mechanisms": ["school funding"],
        "expected_countries": ["United States"],
        "expect_contradiction": True,
        "relevant_papers": [],
    },
    {
        "id": "contested-02",
        "type": "contested",
        "question": "Do studies disagree on neighborhood-effect magnitudes?",
        "expected_mechanisms": ["neighborhood effects"],
        "expected_countries": ["United States"],
        "expect_contradiction": True,
        "relevant_papers": [],
    },
    {
        "id": "contested-03",
        "type": "contested",
        "question": "Is there disagreement about social-capital measurement?",
        "expected_mechanisms": ["social capital"],
        "expected_countries": [],
        "expect_contradiction": True,
        "relevant_papers": [],
    },
    {
        "id": "contested-04",
        "type": "contested",
        "question": "Do wealth-effect estimates conflict across studies?",
        "expected_mechanisms": ["wealth concentration"],
        "expected_countries": [],
        "expect_contradiction": True,
        "relevant_papers": [],
    },
    {
        "id": "contested-05",
        "type": "contested",
        "question": "Where is evidence on labor segmentation contested?",
        "expected_mechanisms": ["labor market segmentation"],
        "expected_countries": [],
        "expect_contradiction": True,
        "relevant_papers": [],
    },
]


def citation_accuracy(answer_claims: list, evidence_spans: list) -> float:
    if not answer_claims:
        return 0.0
    supported = 0
    for claim in answer_claims:
        text = str(claim)
        if any(text[:80] in str(span) or str(span)[:80] in text for span in evidence_spans):
            supported += 1
    return round(supported / len(answer_claims), 4)


def mechanism_coverage(found: list, expected: list) -> float:
    if not expected:
        return 1.0
    found_set = {m.strip().lower() for m in found}
    hits = sum(1 for m in expected if m.strip().lower() in found_set)
    return round(hits / len(expected), 4)


def contradiction_detection(detected: bool, expected: bool) -> float:
    return 1.0 if detected == expected else 0.0


def geographic_specificity(found_countries: list, expected_countries: list) -> float:
    if not expected_countries:
        return 1.0
    found_set = {c.strip().lower() for c in found_countries}
    hits = sum(1 for c in expected_countries if c.strip().lower() in found_set)
    return round(hits / len(expected_countries), 4)


def answer_groundedness(answer_claims: list, evidence_spans: list) -> float:
    return citation_accuracy(answer_claims, evidence_spans)


def run_evaluation(standard_results: list, network_results: list) -> dict:
    rows = []
    for question, standard, network in zip(
        EVALUATION_QUESTIONS, standard_results, network_results
    ):
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "standard": {
                    "citation_accuracy": citation_accuracy(
                        standard.get("claims", []),
                        standard.get("evidence_spans", []),
                    ),
                    "mechanism_coverage": mechanism_coverage(
                        standard.get("mechanisms", []),
                        question["expected_mechanisms"],
                    ),
                    "geographic_specificity": geographic_specificity(
                        standard.get("countries", []),
                        question["expected_countries"],
                    ),
                    "contradiction_detection": contradiction_detection(
                        standard.get("contradiction_found", False),
                        question["expect_contradiction"],
                    ),
                    "answer_groundedness": answer_groundedness(
                        standard.get("claims", []),
                        standard.get("evidence_spans", []),
                    ),
                },
                "network_aware": {
                    "citation_accuracy": citation_accuracy(
                        network.get("claims", []),
                        network.get("evidence_spans", []),
                    ),
                    "mechanism_coverage": mechanism_coverage(
                        network.get("mechanisms", []),
                        question["expected_mechanisms"],
                    ),
                    "geographic_specificity": geographic_specificity(
                        network.get("countries", []),
                        question["expected_countries"],
                    ),
                    "contradiction_detection": contradiction_detection(
                        network.get("contradiction_found", False),
                        question["expect_contradiction"],
                    ),
                    "answer_groundedness": answer_groundedness(
                        network.get("claims", []),
                        network.get("evidence_spans", []),
                    ),
                },
            }
        )
    return {"count": len(rows), "rows": rows}


def write_evaluation_results(results: dict, path: str) -> str:
    import json
    import pathlib

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(target)
