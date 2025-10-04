# Food Price Clustering Project - Technical Documentation

## Project Overview

This project implements a machine learning pipeline for clustering Indonesian cities based on food price patterns across 10 commodities over a 5-year period (2020-2024). The system allows users to configure and execute clustering analyses through an API endpoint.

## Business Objective

Identify cities with similar food price behaviors to:

-   Understand regional market dynamics
-   Identify price stability vs volatility patterns
-   Support policy and supply chain decisions
-   Detect inflation trends and anomalies

## Dataset Characteristics

-   **Temporal Coverage**: 2020-2024 (5 years)
-   **Granularity**: Daily data, weekends excluded (~260-262 days/year)
-   **Spatial Coverage**: 60-70 Indonesian cities
-   **Commodities**: 10 food items
-   **Total Records**: ~78,000-91,000 price observations

## Feature Engineering

From raw time series price data, we extract 3 features per commodity:

1. **Price Average**: Mean price over the period
2. **Coefficient of Variation (CV)**: Volatility measure (std/mean)
3. **Price Trend**: Linear regression slope (price change over time)

**Final Feature Space**: 30 columns (3 features × 10 commodities)

## Clustering Algorithms

Three complementary algorithms will be implemented:

### 1. K-Means Clustering

-   **Use Case**: Fast, efficient for convex clusters
-   **Strengths**: Scalable, interpretable, good baseline
-   **Weaknesses**: Assumes spherical clusters, sensitive to outliers
-   **Implementation**: `sklearn.cluster.KMeans`

### 2. Fuzzy C-Means (FCM)

-   **Use Case**: Soft clustering with membership degrees
-   **Strengths**: Captures cluster uncertainty, identifies boundary cases
-   **Weaknesses**: More computationally expensive
-   **Implementation**: `skfuzzy.cluster.cmeans` or custom implementation
-   **Output**: Membership matrix showing degree of belonging to each cluster

### 3. Spectral Clustering

-   **Use Case**: Non-convex cluster shapes
-   **Strengths**: Can find complex cluster geometries
-   **Weaknesses**: O(n³) complexity, requires tuning affinity parameter
-   **Implementation**: `sklearn.cluster.SpectralClustering`

## Project Development Phases

### Phase 1: Notebook-Based Experimentation (CURRENT PHASE)

**Objective**: Understand data, validate approach, find optimal configurations

**Notebook Structure**:

```
notebooks/
├── 01_preprocessing.ipynb (COMPLETED)
├── 02_feature_engineering.ipynb (IN PROGRESS)
├── 03_clustering_experiments.ipynb (IN PROGRESS)
└── 04_pipeline_validation.ipynb (FUTURE)
```

**Clustering Notebook Sections** (`03_clustering_experiments.ipynb`):

1. Setup & Data Loading
2. Pre-Clustering EDA
3. Data Preparation (Feature Scaling)
4. Optimal K Selection (Elbow + Silhouette)
5. K-Means Implementation & Analysis
6. Fuzzy C-Means Implementation & Analysis
7. Spectral Clustering Implementation & Analysis
8. Algorithm Comparison
9. Deep Dive - Best Performing Algorithm
10. Business Interpretation & Insights

**Key Outputs from Phase 1**:

-   Optimal number of clusters (k)
-   Best performing algorithm
-   Feature importance insights
-   Preprocessing requirements validated
-   Visualization strategies confirmed

### Phase 2: Modularization (AFTER PHASE 1)

**Objective**: Transform notebook code into reusable, testable modules

**Proposed Module Structure**:

```
src/
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py          # Load and validate input data
│   ├── cleaner.py               # Handle nulls, outliers, data types
│   └── consolidator.py          # Data aggregation if needed
├── features/
│   ├── __init__.py
│   └── feature_engineer.py      # Extract 30 features from time series
├── clustering/
│   ├── __init__.py
│   ├── base_clusterer.py        # Abstract base class for all algorithms
│   ├── kmeans_clusterer.py      # K-Means implementation
│   ├── fcm_clusterer.py         # Fuzzy C-Means implementation
│   └── spectral_clusterer.py    # Spectral Clustering implementation
├── analysis/
│   ├── __init__.py
│   ├── metrics.py               # Silhouette, Davies-Bouldin, etc.
│   └── visualizer.py            # Generate plots and visualization data
├── pipeline/
│   ├── __init__.py
│   └── clustering_pipeline.py   # Orchestrate end-to-end workflow
└── utils/
    ├── __init__.py
    └── validators.py            # Input validation helpers
```

### Phase 3: API Development (FINAL PHASE)

**Objective**: Expose clustering pipeline through REST API

**API Structure**:

```
api/
├── main.py           # FastAPI app initialization
├── models.py         # Pydantic schemas for request/response
├── routes.py         # API endpoints
└── dependencies.py   # Shared dependencies, auth if needed
```

**Key Endpoint**: `POST /api/cluster`

**Request Schema**:

```json
{
  "data": "file upload or data array",
  "config": {
    "algorithm": "kmeans | fcm | spectral",
    "n_clusters": 5,
    "fields_to_cluster": ["rice", "oil", "sugar", ...],
    "scaling_method": "standard | minmax | robust",
    "random_state": 42
  }
}
```

**Response Schema**:

```json
{
  "cluster_labels": [0, 1, 2, 1, 0, ...],
  "cluster_centers": [...],
  "metrics": {
    "silhouette_score": 0.65,
    "davies_bouldin_index": 0.85,
    "calinski_harabasz_score": 450.2
  },
  "cluster_profiles": {
    "cluster_0": {
      "size": 15,
      "avg_features": {...},
      "representative_cities": [...]
    }
  },
  "visualizations": {
    "pca_plot_data": {...},
    "cluster_profile_data": {...}
  }
}
```

## Technical Requirements & Best Practices

### Data Preprocessing

```python
# ALWAYS scale features before clustering
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Why: Features have different scales:
# - Price averages: 10,000 - 50,000 IDR
# - Coefficient of Variation: 0.1 - 0.5
# - Trends: -100 to +100
```

### Optimal K Selection

```python
# Use multiple methods to determine optimal k
from sklearn.metrics import silhouette_score

# 1. Elbow Method (inertia)
# 2. Silhouette Score (quality)
# 3. Davies-Bouldin Index (lower is better)

# Test range: k=2 to k=10
# Expected optimal: k=3 to k=8 for 60-70 cities
```

### Evaluation Metrics

Always compute these metrics for comparison:

-   **Silhouette Score**: [-1, 1], higher is better, measures cluster cohesion
-   **Davies-Bouldin Index**: [0, ∞], lower is better, measures cluster separation
-   **Calinski-Harabasz Score**: [0, ∞], higher is better, ratio of between/within cluster variance

### Visualization Strategy

```python
# For 30D data visualization:
# 1. PCA first (reduce to ~10 components, 90%+ variance)
# 2. Then t-SNE or UMAP for 2D visualization

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Two-step dimensionality reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Then visualize
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_pca)
```

### Cluster Interpretation

```python
# Create cluster profiles for business interpretation
cluster_profiles = df_features.groupby('cluster_label').agg({
    'rice_avg': 'mean',
    'rice_cv': 'mean',
    'rice_trend': 'mean',
    # ... all 30 features
}).round(2)

# Add cluster size
cluster_sizes = df_features['cluster_label'].value_counts()

# Identify representative cities (closest to centroid)
from sklearn.metrics.pairwise import euclidean_distances
# Find cities nearest to each cluster center
```

## Code Quality Standards

### For AI Assistants Working on This Project

1. **Modularity First**

    - Each function should have a single, clear responsibility
    - Classes should follow Single Responsibility Principle
    - Use dependency injection for testability

2. **Type Hints Always**

    ```python
    def calculate_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering result."""
        pass
    ```

3. **Comprehensive Docstrings**

    ```python
    def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
        """
        Fit K-Means clustering algorithm.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (model, labels, metrics_dict)

        Raises:
            ValueError: If n_clusters > n_samples
        """
        pass
    ```

4. **Error Handling**

    - Validate inputs at function entry
    - Provide clear, actionable error messages
    - Use custom exceptions for domain-specific errors

    ```python
    class InvalidClusterConfigError(Exception):
        """Raised when cluster configuration is invalid."""
        pass
    ```

5. **Logging Over Print**

    ```python
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting clustering with k=%d", n_clusters)
    ```

6. **Configuration Management**

    - Use YAML or Pydantic for configuration
    - Never hardcode magic numbers
    - Make everything configurable

    ```python
    from pydantic import BaseModel

    class ClusterConfig(BaseModel):
        algorithm: str
        n_clusters: int
        random_state: int = 42
        max_iterations: int = 300
    ```

7. **Testing Requirements**

    - Unit tests for each module
    - Integration tests for pipeline
    - Use pytest fixtures for test data

    ```python
    @pytest.fixture
    def sample_data():
        """Generate synthetic food price data for testing."""
        return np.random.rand(70, 30)  # 70 cities, 30 features
    ```

8. **Performance Considerations**

    - Profile code for bottlenecks (use `cProfile` or `line_profiler`)
    - Cache expensive computations where appropriate
    - For API: Consider async processing for large datasets
    - Set timeouts for clustering operations

9. **Documentation Standards**

    - README.md with setup instructions
    - API documentation (use FastAPI's auto-docs)
    - Inline comments for complex logic only
    - Jupyter notebooks should have markdown explanations

10. **Git Practices**
    - Meaningful commit messages
    - Feature branches
    - Never commit notebooks with outputs (use `.gitignore`)
    - Keep data files out of git (use DVC or similar)

## Expected Cluster Patterns (Hypotheses)

Based on domain knowledge, we might discover:

1. **"Metro Cities"**: High average prices, low volatility, flat trends
2. **"Stable Provincial"**: Medium prices, low CV, flat trends
3. **"Volatile Markets"**: High CV across commodities (remote areas, supply issues)
4. **"Inflation-Hit"**: Positive trends across multiple commodities
5. **"Agricultural Hubs"**: Low prices for certain commodities, varied patterns
6. **"Border/Port Cities"**: Unique patterns based on import/export dynamics

## Performance Benchmarks

Target performance metrics:

-   **Clustering Time**: < 5 seconds for 70 cities, 30 features
-   **API Response Time**: < 10 seconds end-to-end (including visualization)
-   **Memory Usage**: < 500MB for typical workload
-   **Silhouette Score**: > 0.4 (indicates reasonable clustering)

## Security & Validation

### Input Validation

-   File size limits (e.g., max 10MB)
-   File format validation (CSV, Excel)
-   Column name validation
-   Data type checking
-   Range validation (e.g., prices > 0)
-   Date format validation

### API Security

-   Rate limiting
-   Input sanitization
-   File upload scanning
-   Request size limits
-   CORS configuration

## Dependencies

Core libraries:

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scikit-fuzzy>=0.4.2
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.3
fastapi>=0.100.0
pydantic>=2.0.0
uvicorn>=0.23.0
python-multipart>=0.0.6
```

## Future Enhancements

-   [ ] Add DBSCAN for density-based clustering
-   [ ] Implement time-windowed clustering (analyze periods separately)
-   [ ] Add anomaly detection (cities with unusual patterns)
-   [ ] Support for streaming data updates
-   [ ] Interactive visualization dashboard
-   [ ] Export cluster reports to PDF
-   [ ] A/B testing different preprocessing strategies
-   [ ] Model versioning and experiment tracking (MLflow)

## References & Resources

-   Scikit-learn Clustering: https://scikit-learn.org/stable/modules/clustering.html
-   Fuzzy C-Means: https://pythonhosted.org/scikit-fuzzy/
-   UMAP Documentation: https://umap-learn.readthedocs.io/
-   FastAPI Best Practices: https://fastapi.tiangolo.com/

---

**Last Updated**: 2025-10-04  
**Project Status**: Phase 1 - Notebook Experimentation  
**Next Milestone**: Complete clustering experiments and determine optimal approach
