import argparse
import sys
from src.indexer import FeatureIndexer
from src.retriever import ImageRetriever
from src.evaluate import Evaluator

def main():
    parser = argparse.ArgumentParser(description="CBIR System: Indexing, Retrieval, and Evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Indexing
    parser_index = subparsers.add_parser("index", help="Index dataset images")
    parser_index.add_argument("--config", required=True, help="Path to config.json")
    parser_index.add_argument("--images", required=True, help="Path to dataset images folder")
    parser_index.add_argument("--output", required=True, help="Path to save feature index (.json or .npy)")

    # Retrieval
    parser_retrieve = subparsers.add_parser("retrieve", help="Retrieve similar images for a query")
    parser_retrieve.add_argument("--config", required=True, help="Path to config.json")
    parser_retrieve.add_argument("--index", required=True, help="Path to feature index (.json or .npy)")
    parser_retrieve.add_argument("--query", required=True, help="Path to query image")
    parser_retrieve.add_argument("--top_k", type=int, default=5, help="Number of top results to return")

    # Evaluation
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate retrieval performance")
    parser_eval.add_argument("--config", required=True, help="Path to config.json")
    parser_eval.add_argument("--index", required=True, help="Path to feature index (.json or .npy)")
    parser_eval.add_argument("--ground_truth", required=True, help="Path to ground truth JSON file")
    parser_eval.add_argument("--query_folder", required=True, help="Path to query images folder")
    parser_eval.add_argument("--top_k", type=int, default=5, help="Number of top results to consider")

    args = parser.parse_args()

    if args.command == "index":
        print(f"Indexing images in {args.images} ...")
        indexer = FeatureIndexer(config_path=args.config)
        indexer.index_folder(args.images, args.output)
        print(f"Index saved to {args.output}")

    elif args.command == "retrieve":
        print(f"Retrieving similar images for {args.query} ...")
        retriever = ImageRetriever(config_path=args.config, index_path=args.index)
        results = retriever.retrieve(args.query, top_k=args.top_k)
        print("Top results:")
        for i, (img_id, score) in enumerate(results, 1):
            print(f"{i}. {img_id} (score: {score:.4f})")

    elif args.command == "evaluate":
        print(f"Evaluating retrieval performance ...")
        evaluator = Evaluator(
            config_path=args.config,
            index_path=args.index,
            ground_truth_path=args.ground_truth
        )
        metrics = evaluator.evaluate(args.query_folder, top_k=args.top_k)
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
