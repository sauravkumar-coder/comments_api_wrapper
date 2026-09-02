"""
Business semantic feature engineering — 37 hand-built features.

Each feature is a presence flag or count of a business concept, checked via
word-boundary regex.  **Never a function of remark length.**

This must exactly replicate the training notebook's Section 4 logic.
The same functions are called unchanged in the inference pipeline.

Seven major story-element categories:
  1. Visit purpose  (10 sub-categories × 2 features + 1 diversity = 21)
  2. Stakeholder    (4 presence flags + 1 diversity = 5)
  3. Discussion     (present + count = 2)
  4. Response       (present + count = 2)
  5. Action         (present + count = 2)
  6. Outcome        (present + count = 2)
  7. Follow-up      (present + count = 2)
  + story_elements_present composite (1)
  Total = 37
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Keyword lexicons — word-boundary matched
# ═══════════════════════════════════════════════════════════════════════════════

_PURPOSE_CATEGORIES: Dict[str, List[str]] = {
    "review": ["review", "reviewed", "reviewing"],
    "training": ["training", "trained", "demo", "demonstration"],
    "attachment": [
        "attachment", "attach", "display", "planogram",
        "branding", "visibility",
    ],
    "performance": [
        "performance", "target", "achievement",
        "sell out", "sell-out", "sellout",
    ],
    "sales": [
        "sales", "revenue", "offtake", "billing",
        "secondary", "primary",
    ],
    "audit": [
        "audit", "compliance", "check", "inspection", "hygiene",
    ],
    "issue": [
        "issue", "problem", "concern", "challenge", "gap",
        "dead stock", "dead-stock",
    ],
    "complaint": [
        "complaint", "escalation", "service", "warranty", "replacement",
    ],
    "activation": [
        "activation", "scheme", "offer", "promotion",
        "cashback", "exchange",
    ],
    "conversion": [
        "conversion", "convert", "upgrade", "switch",
        "migration", "win back", "win-back",
    ],
}

_STAKEHOLDER_KEYWORDS: Dict[str, List[str]] = {
    "SEC": ["sec", "store executive"],
    "DM": ["dm", "dealer", "distributor"],
    "SM": ["sm", "store manager", "asm"],
    "Team": ["team", "staff", "promoter"],
}

_DISCUSSION_KEYWORDS: List[str] = [
    "discussed", "explained", "analyzed", "identified", "understood",
    "briefed", "presented", "highlighted", "shared", "communicated",
]

_RESPONSE_KEYWORDS: List[str] = [
    "informed", "mentioned", "stated", "highlighted", "confirmed",
    "agreed", "acknowledged", "accepted", "noted", "committed",
]

_ACTION_KEYWORDS: List[str] = [
    "trained", "coached", "escalated", "resolved", "clarified",
    "demonstrated", "corrected", "realigned", "motivated", "guided",
]

_OUTCOME_KEYWORDS: List[str] = [
    "improvement", "root cause", "opportunity", "commitment",
    "increase", "growth", "potential", "insight", "result", "progress",
]

_FOLLOWUP_KEYWORDS: List[str] = [
    "next visit", "follow up", "follow-up", "followup",
    "revisit", "monitor", "scheduled", "plan", "action point",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pattern(keywords: List[str]) -> re.Pattern:
    """Build a compiled word-boundary regex from a keyword list."""
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


# Pre-compile all patterns once at module load time
_PURPOSE_PATTERNS: Dict[str, re.Pattern] = {
    name: _build_pattern(kws) for name, kws in _PURPOSE_CATEGORIES.items()
}
_STAKEHOLDER_PATTERNS: Dict[str, re.Pattern] = {
    name: _build_pattern(kws) for name, kws in _STAKEHOLDER_KEYWORDS.items()
}
_DISCUSSION_PATTERN = _build_pattern(_DISCUSSION_KEYWORDS)
_RESPONSE_PATTERN = _build_pattern(_RESPONSE_KEYWORDS)
_ACTION_PATTERN = _build_pattern(_ACTION_KEYWORDS)
_OUTCOME_PATTERN = _build_pattern(_OUTCOME_KEYWORDS)
_FOLLOWUP_PATTERN = _build_pattern(_FOLLOWUP_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def extract_purpose_features(text: str) -> Dict[str, Any]:
    """Extract 21 visit-purpose features (10×present + 10×count + diversity)."""
    feats: Dict[str, Any] = {}
    purposes_present = 0
    for name, pattern in _PURPOSE_PATTERNS.items():
        count = _count_matches(pattern, text)
        feats[f"purpose_{name}_present"] = int(count > 0)
        feats[f"purpose_{name}_count"] = count
        if count > 0:
            purposes_present += 1
    feats["purpose_diversity"] = purposes_present
    return feats


def extract_stakeholder_features(text: str) -> Dict[str, Any]:
    """Extract 5 stakeholder features (4×present + diversity)."""
    feats: Dict[str, Any] = {}
    stakeholders_present = 0
    for name, pattern in _STAKEHOLDER_PATTERNS.items():
        present = int(_count_matches(pattern, text) > 0)
        feats[f"stakeholder_{name}_present"] = present
        if present:
            stakeholders_present += 1
    feats["stakeholder_diversity"] = stakeholders_present
    return feats


def extract_signal_features(text: str) -> Dict[str, Any]:
    """Extract 8 business-signal features (4 groups × present + count)."""
    feats: Dict[str, Any] = {}
    for name, pattern in [
        ("discussion", _DISCUSSION_PATTERN),
        ("response", _RESPONSE_PATTERN),
        ("action", _ACTION_PATTERN),
        ("outcome", _OUTCOME_PATTERN),
    ]:
        count = _count_matches(pattern, text)
        feats[f"{name}_present"] = int(count > 0)
        feats[f"{name}_count"] = count
    return feats


def extract_followup_features(text: str) -> Dict[str, Any]:
    """Extract 2 follow-up features (present + count)."""
    count = _count_matches(_FOLLOWUP_PATTERN, text)
    return {
        "followup_present": int(count > 0),
        "followup_count": count,
    }


def extract_story_elements_count(feats: Dict[str, Any]) -> int:
    """
    Count how many of the 7 major story categories fired (0-7).

    Categories: purpose, stakeholder, discussion, response, action, outcome,
    follow-up.
    """
    elements = 0
    # Purpose: any purpose sub-category present
    if feats.get("purpose_diversity", 0) > 0:
        elements += 1
    # Stakeholder: any stakeholder present
    if feats.get("stakeholder_diversity", 0) > 0:
        elements += 1
    # Discussion, response, action, outcome, followup
    for key in ["discussion_present", "response_present",
                "action_present", "outcome_present", "followup_present"]:
        if feats.get(key, 0) > 0:
            elements += 1
    return elements


def extract_all_semantic_features(text: str) -> Dict[str, Any]:
    """
    Extract all 37 semantic features from cleaned remark text.

    Parameters
    ----------
    text : str
        Cleaned (lowercased, whitespace-normalised) remark text.

    Returns
    -------
    dict
        37 features matching the names in feature_columns.pkl.
    """
    feats: Dict[str, Any] = {}
    feats.update(extract_purpose_features(text))
    feats.update(extract_stakeholder_features(text))
    feats.update(extract_signal_features(text))
    feats.update(extract_followup_features(text))
    feats["story_elements_present"] = extract_story_elements_count(feats)
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  Explainability engine — Section 10 of the notebook
# ═══════════════════════════════════════════════════════════════════════════════

_CATEGORY_LABELS: Dict[str, Tuple[str, str]] = {
    # key_to_check: (strength_text, missing_text)
    "purpose": (
        "Visit purpose clearly stated",
        "No visit purpose identified",
    ),
    "stakeholder": (
        "Stakeholder identified",
        "No stakeholder mentioned",
    ),
    "discussion": (
        "Business discussion documented",
        "No business discussion captured",
    ),
    "response": (
        "Stakeholder response captured",
        "No stakeholder response captured",
    ),
    "action": (
        "Action taken documented",
        "No action taken documented",
    ),
    "outcome": (
        "Business outcome mentioned",
        "No business outcome mentioned",
    ),
    "followup": (
        "Follow-up plan stated",
        "No follow-up plan",
    ),
}


def generate_explanation(
    feats: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Convert semantic feature flags into human-readable strengths and missing
    elements (Section 10 explainability engine).

    Returns
    -------
    (strengths, missing_elements)
    """
    strengths: List[str] = []
    missing: List[str] = []

    for key, (strength_text, missing_text) in _CATEGORY_LABELS.items():
        if key == "purpose":
            is_present = feats.get("purpose_diversity", 0) > 0
        elif key == "stakeholder":
            is_present = feats.get("stakeholder_diversity", 0) > 0
        else:
            is_present = feats.get(f"{key}_present", 0) > 0

        if is_present:
            strengths.append(strength_text)
        else:
            missing.append(missing_text)

    return strengths, missing


def get_story_elements_flags(feats: Dict[str, Any]) -> Dict[str, bool]:
    """
    Return a dict of the 7 story-element boolean flags for the response schema.
    """
    return {
        "visit_purpose": feats.get("purpose_diversity", 0) > 0,
        "stakeholder": feats.get("stakeholder_diversity", 0) > 0,
        "discussion": feats.get("discussion_present", 0) > 0,
        "response": feats.get("response_present", 0) > 0,
        "action": feats.get("action_present", 0) > 0,
        "outcome": feats.get("outcome_present", 0) > 0,
        "followup": feats.get("followup_present", 0) > 0,
    }
