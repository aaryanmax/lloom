"""Ground-truth concept taxonomy for the synthetic dataset.

Verbatim from the LLooM paper, Appendix C.4 ("we used the following set of
10 Generic concepts and 40 Specific concepts"). Each Specific concept belongs
to exactly one Generic parent; a Generic concept is "implicitly invoked" by
the seed sentences of its Specific children (paper §5.2.1).
"""

from __future__ import annotations

# Ordered: Generic concept name -> list of its 4 Specific children.
GENERIC_TO_SPECIFIC: dict[str, list[str]] = {
    "Election Campaigns": [
        "Fundraising",
        "Candidate Profiles",
        "Political Rallies",
        "Campaign Promises",
    ],
    "Government Policies": [
        "Healthcare Policies",
        "Education Policies",
        "International Relations Policies",
        "Economic Policies",
    ],
    "Political Parties": [
        "Party Platforms",
        "Party Leadership",
        "Party History",
        "Party Factionalism",
    ],
    "Human Rights": [
        "LGBTQ+ Rights",
        "Women's Rights",
        "Racial Equality",
        "Children's Rights",
    ],
    "Immigration": [
        "Border Control Policies",
        "Refugee Policies",
        "Immigration Reform",
        "Illegal Immigration",
    ],
    "Economy": [
        "Taxes",
        "Unemployment",
        "Fiscal Policy",
        "Government Spending",
    ],
    "Healthcare": [
        "Universal Healthcare",
        "Mental Health",
        "Drug Policy",
        "Health Insurance",
    ],
    "Environment": [
        "Climate Change",
        "Renewable Energy",
        "Nature Conservation",
        "Air Pollution",
    ],
    "Foreign Policy": [
        "Trade Agreements",
        "War and Peace",
        "Diplomatic Relations",
        "International Aid",
    ],
    "Gun Control": [
        "Background Checks",
        "Assault Weapons Ban",
        "Gun Control Legislation",
        "Second Amendment Rights",
    ],
}

ALL_GENERIC: list[str] = list(GENERIC_TO_SPECIFIC)
ALL_SPECIFIC: list[str] = [s for children in GENERIC_TO_SPECIFIC.values() for s in children]

SPECIFIC_TO_GENERIC: dict[str, str] = {
    s: g for g, children in GENERIC_TO_SPECIFIC.items() for s in children
}


def parent_of(specific: str) -> str:
    return SPECIFIC_TO_GENERIC[specific]


def parents_of(specifics: list[str]) -> list[str]:
    return sorted({parent_of(s) for s in specifics})
