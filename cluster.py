import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=200, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
    
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Initial centroids
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for _ in range(self.max_iter):
            # Compute distances to all centroids
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0)
                                      for k in range(self.n_clusters)])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Not fitted yet!")
        distances = np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    
    def score(self, X, metric="inertia"):
        if self.centroids is None:
            raise ValueError("Not fitted yet!")
        if metric == "inertia":
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))
            return np.sum(np.min(distances, axis=1))
        else:
            raise ValueError("Unknown metric")