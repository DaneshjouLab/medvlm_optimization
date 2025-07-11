from dataclasses import replace
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, cast
import re
import string
import sys

from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k).

    Numerically stable version defined in
        https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def normalize_text(text: str, should_remove_articles: bool = True) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

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


def quasi_leave_articles_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return (
        1
        if normalize_text(gold, should_remove_articles=False) == normalize_text(pred, should_remove_articles=False)
        else 0
    )


def prefix_exact_match(gold: str, pred: str) -> float:
    """
    The `prefix_exact_match` metric is particularly useful in the zero-shot setting, where the model is
    not given examples of the expected outputs and tends to output more tokens than it should.

    For example, for this zero-shot prompt from BoolQ,

    Passage: Elmendorf Air Force Base (IATA: EDF, ICAO: PAED, FAA LID: EDF) is a United States military facility
    in Anchorage, the largest city in Alaska. Originally known as Elmendorf Field, it became Elmendorf Air Force
    Base after World War II, and in 2010 it merged with nearby Fort Richardson to form Joint Base Elmendorf-Richardson.
    Question: Is there an air force base in anchorage alaska?
    Answer:

    the model could output up to `max_tokens` number of tokens "Yes, Elmendorf" instead of just "Yes".
    """
    if not pred:
        return 0

    return 1 if pred.strip().startswith(gold.strip()) else 0


def quasi_prefix_exact_match(gold: str, pred: str) -> float:
    """
    Same thing as `prefix_exact_match` but we normalize the text before checking if the prefix match.
    """
    if not pred:
        return 0

    return 1 if normalize_text(pred).startswith(normalize_text(gold)) else 0


def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    if ret is None:
        return 0.0

    return ret


def exact_match_indicator(gold: str, pred: str, indicator: str = " ") -> float:
    """
    Exact match, allowing for some preceding context.
    For example, the following two answers are considered matching:
    - Because of x and y, the answer is ## <answer>
    - Given reasons y and z, the answer is ## <answer>
    While the following is considered different from the earlier two
    - Given reasons x and a, the answer is ## <other answer>
    """
    pred = pred.split(indicator)[-1].strip()
    gold = gold.split(indicator)[-1].strip()
    return exact_match(gold, pred)


def final_number_exact_match(gold: str, pred: str) -> float:
    """
    Returns 1 iff the final number in gold and pred match.
    Similar to exact_match_indicator.
    Example:
    - gold = "The answer is 15."
    - pred = "The answer is 15 eggs."
    - Returns 1
    """

    def get_final_number(x: str) -> str:
        matches = re.findall(r"-?[\d,]+(?:.\d+)?", x)
        if not matches:
            return ""
        return matches[-1].replace(",", "")

    return exact_match(get_final_number(gold), get_final_number(pred))


def rouge_score(gold: str, pred: str, rouge_type: str, scorer: rouge_scorer.RougeScorer) -> float:
    scores = scorer.score(gold, pred)
    return scores[rouge_type].fmeasure


def get_rouge_function(rouge_type: str) -> Callable[[str, str], float]:
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    return partial(rouge_score, scorer=scorer, rouge_type=rouge_type)


def bleu_1(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(1, 0, 0, 0))


def chinese_bleu_1(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jieba not installed. Install with: pip install jieba")
    
    gold_tokens = list(jieba.cut(gold))
    pred_tokens = list(jieba.cut(pred))
    return sentence_bleu([gold_tokens], pred_tokens, weights=(1, 0, 0, 0))


def get_chinese_rouge_function(rouge_type: str) -> Callable[[str, str], float]:
    try:
        import jieba
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jieba not installed. Install with: pip install jieba")
    
    def chinese_rouge_score(gold: str, pred: str) -> float:
        gold_tokens = " ".join(jieba.cut(gold))
        pred_tokens = " ".join(jieba.cut(pred))
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        scores = scorer.score(gold_tokens, pred_tokens)
        return scores[rouge_type].fmeasure
    
    return chinese_rouge_score


def cleva_math_result_match(gold: str, pred: str) -> float:
    """
    Exact match that only cares the last math expression.
    Common math expressions are numbers and fractions.
    """
    pattern = r"[-+*/%\.\(\)\d]+"
    matches = re.findall(pattern, pred)
    if matches:
        pred = matches[-1].lstrip(")")
    pred = pred.strip()
    return exact_match(gold, pred)


def bleu_4(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))


def cider(gold: str, pred: str) -> float:
    try:
        from pycocoevalcap.cider.cider import Cider
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pycocoevalcap not installed. Install with: pip install pycocoevalcap")

    cider_evaluator = Cider()
    candidate = {"caption": [pred]}
    reference = {"caption": [gold]}
    average_score, _ = cider_evaluator.compute_score(reference, candidate)
    return average_score


def wer_score(gold: str, pred: str) -> float:
    try:
        from jiwer import wer
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jiwer not installed. Install with: pip install jiwer")

    if not pred:
        return 0
    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    wer_ret = wer(gold, pred)
    return wer_ret


def mer_score(gold: str, pred: str) -> float:
    try:
        from jiwer import mer
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jiwer not installed. Install with: pip install jiwer")

    if not pred:
        return 0

    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    mer_ret = mer(gold, pred)
    return mer_ret


def wip_score(gold: str, pred: str) -> float:
    try:
        from jiwer import wip
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jiwer not installed. Install with: pip install jiwer")

    if not pred:
        return 0

    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    wip_ret = wip(gold, pred)
    return wip_ret


def cer_score(gold: str, pred: str) -> float:
    try:
        from jiwer import cer
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jiwer not installed. Install with: pip install jiwer")

    if not pred:
        return 0

    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    cer_ret = cer(gold, pred)
    assert isinstance(cer_ret, float)
    return cer_ret


def chinese_wer_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jieba not installed. Install with: pip install jieba")

    return wer_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def chinese_mer_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jieba not installed. Install with: pip install jieba")

    return mer_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def chinese_wip_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jieba not installed. Install with: pip install jieba")

    return wip_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def chinese_cer_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError:
        raise ModuleNotFoundError("jieba not installed. Install with: pip install jieba")

    return cer_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def extract_set_from_text(
    set_str: str,
    set_start_str: str = " is ",
    set_separator: str = " and ",
    empty_set_str: str = "Nothing.",
) -> Set[str]:
    """
    Given a string, extract the set of strings implied by that string.
    set_start_str denotes the start of the set
    set_separator denotes the string separating set elements
    empty_set_str is the string which denotes the empty set
    """
    if set_str == empty_set_str:
        return set()
    set_str = set_str.replace(".", "")
    extracted_set = set(set_str.split(set_start_str)[-1].split(set_separator))
    return extracted_set


def extract_gold_pred_sets(gold: str, pred: str) -> Tuple[Set[str], Set[str]]:
    """Extract the set of strings implied by the gold and pred strings"""
    gold_set = extract_set_from_text(gold)
    pred_set = extract_set_from_text(pred.split("\n")[0])
    return gold_set, pred_set


def iou_set_match(gold: str, pred: str) -> float:
    """Compute the intersection over union of the gold and pred sets"""
    gold_set, pred_set = extract_gold_pred_sets(gold, pred)
    if len(gold_set) == 0:
        return float(gold_set == pred_set)
    return len(gold_set.intersection(pred_set)) / len(gold_set.union(pred_set))


def f1_set_match(gold: str, pred: str) -> float:
    """Compute the F1 score of the gold and pred sets"""
    gold_set, pred_set = extract_gold_pred_sets(gold, pred)
    if len(gold_set) == 0:
        return float(gold_set == pred_set)
    true_positives = gold_set.intersection(pred_set)
    return 2 * len(true_positives) / (len(gold_set) + len(pred_set))


def exact_set_match(gold: str, pred: str) -> float:
    """Compute whether the sets generated exactly match"""
    gold_set, pred_set = extract_gold_pred_sets(gold, pred)
    return float(gold_set == pred_set)


def absolute_value_difference(gold: str, pred: str) -> float:
    """Compute the absolute value of the difference between two numbers (provided as strings),
    or 0.0 if invalid input.
    """

    def maybe_int(text: str):
        """Parse int, ignoring commas in numbers."""
        try:
            val = int(text.replace(",", ""))
        except ValueError:
            return 0.0
        return val

    gold_val = maybe_int(gold)
    pred_val = maybe_int(pred)
    return abs(gold_val - pred_val)


def code_eval(gold: Tuple[str, Optional[Dict]], pred: str) -> float:
    """Evaluate Code Correctness on test examples."""
    assert gold[1] is not None
    raise NotImplementedError("Code evaluation requires additional implementation for safe code execution")


def compute_token_totals(history):
    total_input_tokens = 0
    total_output_tokens = 0
    bad_items = []
    for i, item in enumerate(history):
        if "usage" in item and "prompt_tokens" in item["usage"] and "completion_tokens" in item["usage"]:
            total_input_tokens += item["usage"]["prompt_tokens"]
            total_output_tokens += item["usage"]["completion_tokens"]
        else:
            bad_items.append(i)
    print("Total input tokens:", total_input_tokens)
    print("Total output tokens:", total_output_tokens)
    print("Number of bad items:", len(bad_items))
    print("Bad item indices:", bad_items) 