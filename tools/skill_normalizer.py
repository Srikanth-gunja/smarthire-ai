"""Normalize LLM-extracted skill lists before candidate matching.

LLMs sometimes return a category sentence (for example, ``"Languages:
Python, SQL, Java"``) where the schema expects individual skills.  This
module turns those entries into comparable atomic skill names.
"""

from __future__ import annotations

import re

_ALIASES = {
    "scikit learn": "scikit learn",
    "sklearn": "scikit learn",
    "numpy": "numpy",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "lang chain": "langchain",
    "lang graph": "langgraph",
    "data structures and algorithms": "data structures algorithms",
    "dsa": "data structures algorithms",
    "object oriented programming": "oop",
    "relational sql": "sql",
}


def _split_outside_parentheses(text: str) -> list[str]:
    """Split a category entry on list separators while retaining parentheses."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char in ",;" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def normalize_skill_name(skill: str) -> str:
    """Return a comparison key for a single skill name."""
    value = re.sub(r"\([^)]*\)", "", skill).casefold()
    value = re.sub(r"[^a-z0-9+#.]+", " ", value).strip()
    return _ALIASES.get(value, value)


def normalize_skill_list(skills: list[str]) -> list[str]:
    """Expand grouped skill strings into unique, readable atomic entries."""
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in skills:
        if not entry or not entry.strip():
            continue
        # Category labels precede the last colon; the actual list follows it.
        value = entry.rsplit(":", 1)[-1]
        for item in _split_outside_parentheses(value):
            for atom in re.split(r"\s+\band\s+|/", item, flags=re.IGNORECASE):
                atom = re.sub(r"\([^)]*\)", "", atom).strip(" .:-")
                key = normalize_skill_name(atom)
                if key and key not in seen:
                    normalized.append(atom)
                    seen.add(key)
        # Preserve meaningful parenthesized technologies, e.g. Oracle in
        # "Relational SQL (Oracle, PostgreSQL)".
        for group in re.findall(r"\(([^)]*)\)", value):
            if "," not in group:
                continue
            for atom in _split_outside_parentheses(group):
                atom = atom.strip(" .:-")
                key = normalize_skill_name(atom)
                if key and key not in seen:
                    normalized.append(atom)
                    seen.add(key)
    return normalized


def skills_match(left: str, right: str) -> bool:
    """Compare two skill labels with light aliases and safe containment."""
    left_key, right_key = normalize_skill_name(left), normalize_skill_name(right)
    if not left_key or not right_key:
        return False
    return (
        left_key == right_key
        or (len(left_key) >= 4 and left_key in right_key)
        or (len(right_key) >= 4 and right_key in left_key)
    )
