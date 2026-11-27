"""Zero-external-lexicon structural and pragmatic surface cues (EN + ZH)."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

STRUCT_DIM = 20

_EN_NEG = frozenset(
    {
        "no", "not", "never", "neither", "nor", "none", "nothing", "nowhere",
        "hardly", "barely", "scarcely", "seldom", "rarely", "without", "lack",
    }
)
_EN_ADV = frozenset(
    {
        "but", "however", "yet", "although", "though", "whereas", "while",
        "nevertheless", "nonetheless", "despite", "except", "instead", "rather",
    }
)
_EN_HEDGE = frozenset(
    {
        "maybe", "perhaps", "might", "may", "could", "seem", "seems", "seemed",
        "appear", "appears", "sort", "kind", "somewhat", "slightly", "allegedly",
        "supposedly", "probably", "possibly",
    }
)
_EN_INTENS = frozenset(
    {
        "very", "extremely", "totally", "completely", "utterly", "absolutely",
        "literally", "terribly", "awfully", "incredibly", "deeply", "highly",
        "so", "too", "really", "super",
    }
)
_EN_SELF = frozenset({"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"})
_EN_YOU = frozenset({"you", "your", "yours", "yourself", "yourselves"})

_ZH_NEG = ("不", "没", "没有", "非", "别", "莫", "勿", "未", "毫无", "并非")
_ZH_ADV = ("但是", "但", "然而", "不过", "却", "虽然", "虽", "尽管", "反而", "倒不如")
_ZH_HEDGE = ("可能", "也许", "似乎", "好像", "大概", "或许", "有点", "略微", "看似")
_ZH_INTENS = ("非常", "极其", "简直", "彻底", "完全", "太", "真", "特别", "十分", "格外")
_ZH_SELF = ("我", "我们", "咱", "咱们", "本人")
_ZH_YOU = ("你", "你们", "您")

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CLAUSE_RE = re.compile(r"[,;:—，；：]+")
_ELONG_RE = re.compile(r"(.)\1{2,}")


def _is_cjk_dominant(text: str) -> bool:
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk >= max(2, int(0.15 * len(text)))


def extract_struct_vector(text: str) -> np.ndarray:
    """Extract 20 zero-external-lexicon structural/pragmatic cues from raw text."""
    s = str(text or "")
    n_char = max(len(s), 1)
    feat = np.zeros(STRUCT_DIM, dtype=np.float32)

    excl = s.count("!") + s.count("！")
    quest = s.count("?") + s.count("？")
    mixed_eq = 1.0 if (excl > 0 and quest > 0) else 0.0
    ellip = s.count("...") + s.count("…")
    quotes = (
        s.count('"') // 2
        + s.count("“")
        + s.count("「")
        + s.count("『")
    )
    upper_chars = sum(1 for c in s if "A" <= c <= "Z")
    alpha_chars = sum(1 for c in s if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    upper_ratio = upper_chars / max(alpha_chars, 1)
    elong = len(_ELONG_RE.findall(s))
    emoji_cnt = sum(1 for c in s if 0x1F300 <= ord(c) <= 0x1FAFF)
    clause_cnt = 1.0 + len(_CLAUSE_RE.findall(s))

    feat[0] = np.log1p(n_char)
    feat[1] = np.log1p(excl)
    feat[2] = np.log1p(quest)
    feat[3] = mixed_eq
    feat[4] = np.log1p(ellip)
    feat[5] = np.log1p(quotes)
    feat[6] = upper_ratio
    feat[7] = np.log1p(elong)
    feat[8] = np.log1p(emoji_cnt)
    feat[9] = np.log1p(clause_cnt)

    if _is_cjk_dominant(s):
        try:
            import jieba  # type: ignore

            toks = [t.strip() for t in jieba.lcut(s) if t.strip()]
        except Exception:
            toks = list(s.strip())
        n_tok = max(len(toks), 1)
        tok_set = toks
        neg = sum(1 for t in tok_set if t in _ZH_NEG)
        adv = sum(1 for t in tok_set if t in _ZH_ADV)
        hdg = sum(1 for t in tok_set if t in _ZH_HEDGE)
        itn = sum(1 for t in tok_set if t in _ZH_INTENS)
        slf = sum(1 for t in tok_set if t in _ZH_SELF)
        you = sum(1 for t in tok_set if t in _ZH_YOU)
    else:
        words = _WORD_RE.findall(s.lower())
        toks = words
        n_tok = max(len(words), 1)
        neg = sum(1 for w in words if w in _EN_NEG or w.endswith("n't"))
        adv = sum(1 for w in words if w in _EN_ADV)
        hdg = sum(1 for w in words if w in _EN_HEDGE)
        itn = sum(1 for w in words if w in _EN_INTENS)
        slf = sum(1 for w in words if w in _EN_SELF)
        you = sum(1 for w in words if w in _EN_YOU)

    n_tok_f = float(max(len(toks), 1))
    ttr = len(set(toks)) / n_tok_f if toks else 0.0
    mean_tok_len = sum(len(t) for t in toks) / n_tok_f if toks else 0.0
    # Split-half contrast flag: presence of adversative pivot inside interior of sentence
    has_mid_pivot = 1.0 if adv > 0 and clause_cnt >= 2.0 else 0.0

    feat[10] = np.log1p(len(toks))
    feat[11] = ttr
    feat[12] = np.log1p(mean_tok_len)
    feat[13] = neg / n_tok_f
    feat[14] = adv / n_tok_f
    feat[15] = hdg / n_tok_f
    feat[16] = itn / n_tok_f
    feat[17] = slf / n_tok_f
    feat[18] = you / n_tok_f
    feat[19] = has_mid_pivot
    return feat


def extract_struct_matrix(texts: Sequence[str]) -> np.ndarray:
    """Batch-extract (N, 20) float32 structural feature matrix."""
    if not texts:
        return np.zeros((0, STRUCT_DIM), dtype=np.float32)
    return np.vstack([extract_struct_vector(t) for t in texts]).astype(np.float32)
