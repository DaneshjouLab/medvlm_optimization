# This source file is part of the Daneshjou Lab project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

import re
import string
from nltk.metrics.scores import f_measure


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def normalize_text(text: str, should_remove_articles: bool = True) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    normalized_text = remove_punc(lower(text))
    if should_remove_articles:
        normalized_text = remove_articles(normalized_text)
    return white_space_fix(normalized_text)


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def quasi_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    if ret is None:
        return 0.0

    return ret