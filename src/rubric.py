"""Keyword baseline for the DEI-language rubric.

Every cue below comes from the scoring factors in the project brief:
  Celebrator   - ownership + action + naming stakeholders / the issue
  Tolerator    - compliance and legal boilerplate
  Depreciator  - DEI framed as risk, or withdrawal language

This is the NON-AI baseline. The embedding classifier is later compared to it.
"""
import html
import re
from dataclasses import dataclass, field

CLASSES = ["Celebrator", "Tolerator", "Depreciator"]

# (regex, class). Edit these lists to tune the rubric; tests cover the examples.
CUES = [
    # Celebrator: ownership, action, stakeholders, naming the issue
    (r"\bso must we\b", "Celebrator"),
    (r"\bwe can do\b", "Celebrator"),
    (r"\bwe invest(?:ed)? in\b", "Celebrator"),
    (r"\bresponsibilit(?:y|ies)\b", "Celebrator"),
    (r"\bmust do their part\b", "Celebrator"),
    (r"\belected officials\b", "Celebrator"),
    (r"\bcommunity activists\b", "Celebrator"),
    (r"\bracial (?:inequality|equity|justice|wealth gap)\b", "Celebrator"),
    (r"\bobjective analysis\b", "Celebrator"),
    # Tolerator: compliance focus, passive commitment, boilerplate
    (r"\bequal opportunity employer\b", "Tolerator"),
    (r"\bdoes not discriminate\b", "Tolerator"),
    (r"\bcompl(?:y|ying|iance)\b", "Tolerator"),
    (r"\bapplicable\b", "Tolerator"),
    (r"\blaws?\b", "Tolerator"),
    (r"\bfederal and state\b", "Tolerator"),
    (r"\bfair employment\b", "Tolerator"),
    # Depreciator: risk framing and withdrawal
    (r"\bmight not resonate\b", "Depreciator"),
    (r"\bmay not resonate\b", "Depreciator"),
    (r"\bdamage\b", "Depreciator"),
    (r"\breputation\b", "Depreciator"),
    (r"\blitigation\b", "Depreciator"),
    (r"\bbacklash\b", "Depreciator"),
    (r"\bretire[ds]?\b", "Depreciator"),
    (r"\bdisappointed\b", "Depreciator"),
    (r"\bdissolv(?:e|ed|ing)\b", "Depreciator"),
    (r"\bdiscontinu(?:e|ed|ing)\b", "Depreciator"),
]


@dataclass
class Result:
    label: str                      # a class name, or "Unclear" when no cues are found
    score: float                    # (celebrator - depreciator) / total cues, in [-1, 1]
    counts: dict = field(default_factory=dict)
    matches: list = field(default_factory=list)   # (start, end, cue_text, class)


def score_passage(text: str) -> Result:
    """Classify one passage by counting rubric cues."""
    counts = {c: 0 for c in CLASSES}
    matches = []
    for pattern, cls in CUES:
        for m in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            counts[cls] += 1
            matches.append((m.start(), m.end(), m.group(0), cls))
    matches.sort()
    total = sum(counts.values())
    if total == 0:
        return Result("Unclear", 0.0, counts, matches)
    score = (counts["Celebrator"] - counts["Depreciator"]) / total
    # On a tie, prefer Depreciator, then Tolerator, then Celebrator, so a tie never
    # silently counts as the most favorable reading.
    best = max(counts.values())
    label = next(c for c in ["Depreciator", "Tolerator", "Celebrator"] if counts[c] == best)
    return Result(label, score, counts, matches)


def highlight_html(text: str, matches: list) -> str:
    """Return HTML with each matched cue wrapped in a labeled <mark>. Text is escaped."""
    out, pos = [], 0
    for start, end, cue, cls in matches:
        if start < pos:          # skip overlapping matches
            continue
        out.append(html.escape(text[pos:start]))
        out.append(
            f'<mark style="background:#ffe9a8;padding:0 2px;border-radius:3px">'
            f'{html.escape(text[start:end])}'
            f'<sup style="font-size:0.65em"> {cls[:3]}</sup></mark>'
        )
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)
