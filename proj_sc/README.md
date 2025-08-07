# Content-Based Image Retrieval (CBIR) System

This project implements a modular Content-Based Image Retrieval (CBIR) system using DWT Entropy, POPMV, and Color Statistics feature sets.

## Features
- **Preprocessing:** Flexible, config-driven image preprocessing (resize, grayscale, normalization, noise reduction).
- **Feature Extraction:**
  - POPMV (Peak Oriented Octal Pattern Derived Majority Voting)
  - DWT Entropy (Discrete Wavelet Transform + Entropy)
  - Color Statistics (mean, std, skewness for RGB/HSV)
- **Feature Fusion:** Concatenates and normalizes feature vectors.
- **Indexing:** Batch feature extraction and index creation for large datasets.
- **Retrieval:** Fast similarity search using Euclidean or Cosine distance.
- **Evaluation:** Standard metrics (Precision@K, Recall@K, MAP) for benchmarking.
- **Experimentation:** Jupyter notebook for interactive testing and visualization.

## Project Structure
```
proj_sc/
  config.json                # Preprocessing and system config
  requirements.txt           # Python dependencies
  main.py                    # (Entry point, optional)
  README.md                  # Project documentation
  notebooks/
    experimentation.ipynb    # Interactive CBIR demo and testing
  src/
    preprocessing.py         # Image preprocessing logic
    fusion.py                # Feature fusion logic
    indexer.py               # Feature indexing
    retriever.py             # Image retrieval
    evaluate.py              # Evaluation metrics
    feature_extractors/
      __init__.py
      popmv.py               # POPMV feature extraction
      dwt_entropy.py         # DWT entropy feature extraction
      color_stats.py         # Color statistics extraction
  utils/
    image_io.py              # Image loading/saving/display helpers
    metrics.py               # Evaluation metric functions
    helpers.py               # Miscellaneous utilities
```

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your dataset images in `data/images/` and query images in `data/query/`.
3. Edit `config.json` to adjust preprocessing settings as needed.

## Usage
- **Indexing:**
  Run feature extraction and indexing on your dataset:
  ```python
  from src.indexer import FeatureIndexer
  indexer = FeatureIndexer(config_path='config.json', ...)
  indexer.index_folder('data/images', 'data/features/features.json')
  ```
- **Retrieval:**
  Retrieve similar images for a query:
  ```python
  from src.retriever import ImageRetriever
  retriever = ImageRetriever(config_path='config.json', index_path='data/features/features.json', ...)
  results = retriever.retrieve('data/query/query1.jpg', top_k=5)
  print(results)
  ```
- **Evaluation:**
  Evaluate retrieval performance:
  ```python
  from src.evaluate import Evaluator
  evaluator = Evaluator(config_path='config.json', index_path='data/features/features.json', ground_truth_path='data/query/ground_truth.json', ...)
  evaluator.evaluate('data/query', top_k=5)
  ```
- **Experimentation:**
  Use the Jupyter notebook in `notebooks/experimention.ipynb` for interactive testing and visualization.

## References
- [Original Paper: Content Based Image Retrieval Based On DWT Entropy And Popmv Oriented Feature Sets](documents/Content_Based_Image_Retrieval_Based_On_DWT_Entropy_And_Popmv_Oriented_Feature_Sets.pdf)

---

For more details, see the code comments and the notebook. Contributions and suggestions are welcome!
