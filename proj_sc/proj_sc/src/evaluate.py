import json
from typing import List, Dict, Optional
import numpy as np
from src.retriever import ImageRetriever
from utils.metrics import precision_at_k, recall_at_k, mean_average_precision

class Evaluator:
    def __init__(self, 
                 config_path: str, 
                 index_path: str, 
                 ground_truth_path: str,
                 popmv_params: Optional[dict] = None,
                 dwt_params: Optional[dict] = None,
                 color_params: Optional[dict] = None,
                 fusion_params: Optional[dict] = None,
                 distance_metric: str = 'euclidean'):
        """
        Initialize the evaluator with all necessary parameters and ground truth.
        """
        self.retriever = ImageRetriever(
            config_path=config_path,
            index_path=index_path,
            popmv_params=popmv_params,
            dwt_params=dwt_params,
            color_params=color_params,
            fusion_params=fusion_params,
            distance_metric=distance_metric
        )
        self.ground_truth = self._load_ground_truth(ground_truth_path)

    def _load_ground_truth(self, path: str) -> Dict[str, List[str]]:
        """
        Load ground truth from a JSON file mapping query image IDs to lists of relevant image IDs.
        """
        with open(path, 'r') as f:
            return json.load(f)

    def evaluate(self, query_folder: str, top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval performance over all queries in the folder.
        :param query_folder: Folder containing query images.
        :param top_k: Number of top results to consider for metrics.
        :return: Dictionary of evaluation metrics.
        """
        precisions, recalls, aps = [], [], []
        for query_img in self.ground_truth:
            query_path = f"{query_folder}/{query_img}"
            relevant = set(self.ground_truth[query_img])
            results = self.retriever.retrieve(query_path, top_k=top_k)
            retrieved = [img_id for img_id, _ in results]
            precisions.append(precision_at_k(retrieved, relevant, k=top_k))
            recalls.append(recall_at_k(retrieved, relevant, k=top_k))
            aps.append(mean_average_precision(retrieved, relevant))
        metrics = {
            'mean_precision@k': float(np.mean(precisions)),
            'mean_recall@k': float(np.mean(recalls)),
            'MAP': float(np.mean(aps))
        }
        print("Evaluation Results:", metrics)
        return metrics

# Example usage:
# evaluator = Evaluator(
#     config_path='../config.json',
#     index_path='../data/features/features.json',
#     ground_truth_path='../data/query/ground_truth.json',
#     popmv_params={'window_size': 3},
#     dwt_params={'wavelet': 'haar', 'level': 1},
#     color_params={'color_space': 'RGB'},
#     fusion_params={'normalize': True, 'norm_type': 'l2'},
#     distance_metric='euclidean'
# )
# evaluator.evaluate('../data/query', top_k=5)
