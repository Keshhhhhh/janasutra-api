"""
AI Service — Pluggable analysis module for JanaSutra.

Current: Smart keyword-based classification from user description text.
Future:  Swap in Gemini API or custom-trained model by replacing
         analyze_complaint() internals. The function signature stays the same.
"""

import random
import re


# ── Keyword rules: ordered by specificity (most specific first) ──────────
# Each rule: (category, severity, list_of_keyword_patterns)
# Patterns are compiled as case-insensitive regex

CLASSIFICATION_RULES = [
    # Roads
    ("pothole",        3, [r"\bpothole", r"\bpot\s*hole", r"\bpit\b.*road", r"\bcrater\b.*road"]),
    ("road_damage",    2, [r"\broad\s*(damage|crack|broken|break|crumble|deteriorat|cave|sink)",
                           r"\bcrack\b.*road", r"\bbroken\s*road", r"\basphalt",
                           r"\broad\b.*crack", r"\broad\b.*crumbl", r"\broad\b.*broken",
                           r"\bpave?ment\b.*broken", r"\btarring", r"\bspeed\s*breaker\b.*broken"]),

    # Water & Drainage
    ("water_leak",     2, [r"\bwater\s*leak", r"\bleak\b.*pipe", r"\bleak\b.*water",
                           r"\bburst\b.*pipe", r"\bpipe\s*burst", r"\bwater\s*pipe\b.*broken",
                           r"\bflooding", r"\bwater\s*log"]),
    ("drainage",       2, [r"\bdrain", r"\bmanhole", r"\bstorm\s*water", r"\bclog",
                           r"\bblock\b.*drain", r"\boverflow\b.*drain"]),
    ("sewage",         3, [r"\bsewage", r"\bsewer", r"\bseptic", r"\bwaste\s*water\b.*overflow"]),

    # Electricity
    ("streetlight",    1, [r"\bstreet\s*light", r"\bstreetlight", r"\blamp\s*post",
                           r"\blight\s*pole", r"\blight\b.*not\s*work", r"\bno\s*light",
                           r"\bdark\s*road", r"\bbulb\b.*fused?\b", r"\bbulb\b.*broken"]),
    ("electricity",    2, [r"\belectric", r"\bpower\s*cut", r"\btransformer",
                           r"\bwire\b.*hang", r"\bwire\b.*dang", r"\bspark",
                           r"\bshort\s*circuit", r"\bpower\s*line"]),

    # Waste
    ("garbage",        2, [r"\bgarbage", r"\btrash", r"\brubbish", r"\bwaste\b(?!\s*water)",
                           r"\bdump", r"\blitter", r"\brefuse\b.*pile", r"\bdebris",
                           r"\bdirty\b", r"\bfoul\s*smell", r"\bstink", r"\bstench"]),
    ("waste",          1, [r"\bwaste\b", r"\bbio\s*waste", r"\brecycl"]),

    # Trees & Green
    ("tree_fall",      3, [r"\btree\b.*fall", r"\btree\b.*block", r"\bfallen\s*tree",
                           r"\buprooted", r"\bbranch\b.*fall", r"\bbranch\b.*block",
                           r"\btree\b.*danger", r"\btree\b.*lean"]),

    # Traffic
    ("traffic_signal", 2, [r"\btraffic\s*signal", r"\btraffic\s*light", r"\bsignal\b.*not\s*work",
                           r"\bred\s*light\b.*broken", r"\bsignal\b.*broken",
                           r"\btraffic\s*sign"]),
]

# Pre-compile patterns
_COMPILED_RULES = []
for cat, sev, patterns in CLASSIFICATION_RULES:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    _COMPILED_RULES.append((cat, sev, compiled))


def classify_text(description: str) -> tuple[str, int]:
    """
    Classify a complaint description into (category, severity).
    Uses keyword matching with priority ordering.
    Returns ('other', 2) if no pattern matches.
    """
    if not description:
        return ("other", 2)

    text = description.lower().strip()

    # Score each category: count how many patterns match
    best_cat = "other"
    best_sev = 2
    best_score = 0

    for cat, sev, compiled_patterns in _COMPILED_RULES:
        score = sum(1 for p in compiled_patterns if p.search(text))
        if score > best_score:
            best_score = score
            best_cat = cat
            best_sev = sev

    return (best_cat, best_sev)


def analyze_complaint(description: str, image_path: str = None) -> dict:
    """
    Analyze a complaint using its text description (and optionally image).

    Current implementation: keyword-based classification.
    Future: replace internals with Gemini API call or custom CNN+NLP model.

    Returns:
        {
            "category": str,       # e.g. "pothole", "garbage", "water_leak"
            "severity": int,       # 1 (low), 2 (medium), 3 (high)
            "description": str,    # AI-generated summary
            "is_authentic": bool   # forgery detection result
        }
    """
    category, severity = classify_text(description)

    # Generate a contextual AI description
    category_labels = {
        "pothole": "a road surface pothole",
        "road_damage": "damaged road infrastructure",
        "water_leak": "a water supply leak or pipe issue",
        "drainage": "a drainage or stormwater blockage",
        "sewage": "a sewage overflow or contamination issue",
        "streetlight": "a non-functional or damaged streetlight",
        "electricity": "an electrical infrastructure issue",
        "garbage": "illegal waste dumping or garbage accumulation",
        "waste": "improper waste disposal",
        "tree_fall": "a fallen or dangerous tree",
        "traffic_signal": "a malfunctioning traffic signal",
        "other": "a civic issue requiring attention",
    }

    ai_desc = f"Detected: {category_labels.get(category, 'a civic issue')}."

    return {
        "category": category,
        "severity": severity,
        "description": ai_desc,
        "is_authentic": True  # Forgery detection stub — always passes for now
    }


# ── Legacy function name for backward compatibility ──────────────────────
def analyze_image(image_path: str, description: str = "") -> dict:
    """Backward-compatible wrapper. Prefer analyze_complaint() for new code."""
    return analyze_complaint(description=description, image_path=image_path)


# ── Text embedding stubs (for deduplication) ─────────────────────────────

def get_text_embedding(text: str) -> list[float]:
    """
    Returns a 384-dim pseudo-vector for deduplication.
    Replace with IndicSBERT / sentence-transformers later.
    """
    random.seed(hash(text))
    return [random.uniform(-1.0, 1.0) for _ in range(384)]


def compute_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1) ** 0.5
    norm_b = sum(b * b for b in vec2) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)
