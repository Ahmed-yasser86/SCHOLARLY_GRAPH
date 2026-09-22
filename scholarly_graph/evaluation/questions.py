"""Evaluation question set and metric computations."""

from __future__ import annotations

QUESTION_TYPES = ("mechanism", "geographic", "temporal", "contested")

EVALUATION_QUESTIONS: list = [
    {
        "id": f"mechanism-{i:02d}",
        "type": "mechanism",
        "question": question,
        "expected_mechanisms": ["educational inequality"],
        "expected_countries": ["united states"],
        "expect_contradiction": False,
    }
    for i, question in enumerate(
        [
            "How does educational inequality mediate economic inequality and intergenerational mobility?",
            "What role does residential segregation play between inequality and mobility?",
            "How does social capital transmit advantage across generations?",
            "Does wealth concentration reduce intergenerational mobility?",
            "How does labor market segmentation limit upward mobility?",
            "Do political institutions mediate the Great Gatsby Curve?",
            "How does school funding inequality affect child earnings?",
            "Does neighborhood quality change adult earnings?",
            "How do professional networks shape hiring for low-income youth?",
            "Does inheritance explain cross-country mobility differences?",
            "How does early childhood investment affect mobility?",
            "Does private schooling widen mobility gaps?",
            "How do property taxes transmit inequality into school quality?",
            "Does college completion mediate parental income effects?",
            "How does credential licensing segment labor markets?",
            "Does redistribution weaken the inequality-mobility link?",
            "How does healthcare access affect intergenerational outcomes?",
            "Does tutoring widen educational inequality?",
            "How does vocational training shape German mobility patterns?",
            "Does minimum wage policy affect mobility?",
        ],
        start=1,
    )
] + [
    {
        "id": f"geographic-{i:02d}",
        "type": "geographic",
        "question": question,
        "expected_mechanisms": ["educational inequality"],
        "expected_countries": ["united states", "sweden"],
        "expect_contradiction": False,
    }
    for i, question in enumerate(
        [
            "How does the education mechanism differ between the United States and Sweden?",
            "Compare neighborhood effects in the United States and Denmark.",
            "Is wealth concentration stronger in Brazil or the United States?",
            "How do Nordic public schools change mobility relative to the US?",
            "Compare informal labor markets in Brazil and Egypt.",
            "Does social capital work differently in Germany and France?",
            "How does school funding compare across the US and Finland?",
            "Are tutoring effects larger in Egypt or Norway?",
            "Compare inheritance regimes in France and the United States.",
            "How does redistribution differ between Denmark and Brazil?",
            "Do neighborhood effects replicate outside the United States?",
            "Is the Great Gatsby Curve visible in developing economies?",
            "Compare vocational training in Germany and Egypt.",
            "How does property-tax school funding differ internationally?",
            "Does early childhood policy explain Nordic mobility?",
        ],
        start=1,
    )
] + [
    {
        "id": f"temporal-{i:02d}",
        "type": "temporal",
        "question": question,
        "expected_mechanisms": ["economic inequality"],
        "expected_countries": ["united states"],
        "expect_contradiction": False,
    }
    for i, question in enumerate(
        [
            "How has the inequality-mobility literature evolved since 2000?",
            "Did mechanism research shift after Piketty's Capital?",
            "How have Nordic mobility estimates changed over time?",
            "Has US educational inequality grown since the 1990s?",
            "How did administrative tax data change mobility research?",
            "Have neighborhood-effect estimates changed with new methods?",
            "How has social-capital measurement evolved?",
            "Did quasi-experimental designs change mechanism conclusions?",
            "How has developing-country evidence grown over time?",
            "What do recent studies add to the Great Gatsby Curve?",
        ],
        start=1,
    )
] + [
    {
        "id": f"contested-{i:02d}",
        "type": "contested",
        "question": question,
        "expected_mechanisms": ["educational inequality"],
        "expected_countries": ["united states"],
        "expect_contradiction": True,
    }
    for i, question in enumerate(
        [
            "Where does the literature disagree about school funding effects?",
            "Do studies disagree on neighborhood-effect magnitudes?",
            "Is there disagreement about social-capital measurement?",
            "Do wealth-effect estimates conflict across studies?",
            "Where is evidence on labor segmentation contested?",
        ],
        start=1,
    )
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
