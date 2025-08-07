import numpy as np
from typing import List, Set

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute precision at k.
    :param retrieved: List of retrieved image IDs.
    :param relevant: Set of relevant image IDs.
    :param k: Number of top results to consider.
    :return: Precision at k.
    """
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = [img for img in retrieved_k if img in relevant]
    return len(relevant_retrieved) / k

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute recall at k.
    :param retrieved: List of retrieved image IDs.
    :param relevant: Set of relevant image IDs.
    :param k: Number of top results to consider.
    :return: Recall at k.
    """
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = [img for img in retrieved_k if img in relevant]
    return len(relevant_retrieved) / len(relevant)

def mean_average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Compute mean average precision (AP) for a single query.
    :param retrieved: List of retrieved image IDs.
    :param relevant: Set of relevant image IDs.
    :return: Average precision.
    """
    if not relevant:
        return 0.0
    ap = 0.0
    hit_count = 0
    for i, img_id in enumerate(retrieved):
        if img_id in relevant:
            hit_count += 1
            ap += hit_count / (i + 1)
    return ap / len(relevant) if relevant else 0.0

def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Compute mean reciprocal rank (MRR) for a single query.
    :param retrieved: List of retrieved image IDs.
    :param relevant: Set of relevant image IDs.
    :return: Reciprocal rank.
    """
    for i, img_id in enumerate(retrieved):
        if img_id in relevant:
            return 1.0 / (i + 1)
    return 0.0
