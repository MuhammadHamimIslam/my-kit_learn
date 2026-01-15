# my-kit_learn

Machine learning algorithms implemented **from scratch** in pure Python (no scikit-learn, TensorFlow, PyTorch, etc.).

This is a personal learning project to deeply understand how classic ML algorithms work under the hood.

## Features

- Pure NumPy-based implementations
- Modular design (one main class/algorithm per file)
- Clean, readable code with comments
- MIT licensed – feel free to use, modify, and learn from it

## Implemented Algorithms

| Module              | Algorithms / Models Implemented                  | Status     |
|---------------------|--------------------------------------------------|------------|
| `preprocessor.py`   | Scaling, encoding, missing value handling, etc.  | ✅         |
| `linear_model.py`   | Linear Regression, Logistic Regression, Ridge    | ✅         |
| `knn.py`            | K-Nearest Neighbors (classification & regression)| ✅         |
| `tree.py`           | Decision Trees (CART-style)                      | ✅         |
| `cluster.py`        | K-Means, possibly hierarchical clustering        | ✅         |
| `neural_net.py`     | Feedforward Neural Network with backpropagation  | ✅ (recent) |

More algorithms (SVM, Naive Bayes, Random Forest, PCA, etc.) are planned.

## Requirements

```text
numpy
matplotlib  # for visualization examples (optional)
```

Install via:

```bash
pip install -r requirements.txt
```

## Usage Example

```python
# Simple linear regression example (check linear_model.py for full API)
from linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

print("Prediction for x=6:", model.predict(np.array([[6]])))
```

## Project Goals

- Understand mathematics and implementation details of each algorithm
- Write clean, vectorized, and efficient code using NumPy
- Compare results with scikit-learn (for validation only)
- Serve as a personal reference and teaching resource

## Roadmap

- Add more algorithms (SVM, ensemble methods, dimensionality reduction)
- Implement proper train/test split and cross-validation utilities
- Add visualization helpers (decision boundaries, learning curves)
- Create Jupyter notebook demos for each algorithm
- Write comprehensive docstrings and usage examples

## License

MIT License  
Copyright (c) 2025-2026 Muhammad Hamim Islam

See the [LICENSE](LICENSE) file for full details.