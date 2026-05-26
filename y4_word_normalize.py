from __future__ import annotations

import string

HYPHEN_LIKE = "-\u2010\u2011\u2012\u2013\u2014\u2212"

_HYPHEN_LIKE_SET = frozenset(HYPHEN_LIKE)


def normalize_y4_words(text: str | None) -> list[str]:
    """
    Match step1b / step3 Y4 word counting:
    1) replace hyphen-like characters with ASCII space (so coffee-cup → two tokens),
    2) remove remaining ASCII punctuation (string.punctuation),
    3) split on whitespace.
    """
    s = text if text is not None else ""
    for ch in HYPHEN_LIKE:
        s = s.replace(ch, " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s.split()


def y4_has_forbidden_punctuation(text: str) -> bool:
    """
    Y4 surface check: disallow punctuation except hyphen-like chars (allowed for compounds).
    """
    for ch in text:
        if ch in string.punctuation and ch not in _HYPHEN_LIKE_SET:
            return True
    return False
