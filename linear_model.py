import numpy as np

class LinearRegression:
    """
    ...
    """
    def __init__(self, n_iter=100, learning_rate=0.01):
        self.n_iter = n_iter
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_pred = (X @ self.weights) + self.bias # predict
            
            # Calculate the gradient
            dW = (1 / n_samples) * (X.T @ (y_pred - y))
            dB = (1 / n_samples) * np.sum(y_pred - y)
            
            # update gradient
            self.weights -= dW * self.lr
            self.bias -= dB * self.lr
        print("Fitted")

    def predict(self, X):
        return (X @ self.weights) + self.bias # predict
    
    def score(self, X, y, metric="MSE"):
        """
        Args -> X: test data, y: label
        returns -> mean squared error, root mean squared error, R² score
        """
        y_pred = X @ self.weights + self.bias
        
        error = y_pred - y
        if metric == "MSE":
            return np.mean(error ** 2)
        elif metric == "RMSE":
            return np.sqrt(np.mean(error ** 2))
        elif metric == "r2":
            return 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))
        else:
            raise ValueError("Not a familiar metric!")

class StandardScaler:
    """
    ...
    """
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if (self.std == 0).any():
            warnings.warn("Your all values are constant. So std is 0")
    
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted yet! Call fit or fit_transform first.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if (self.std == 0).any():
            warnings.warn("Your all values are constant. So std is 0")

        return (X - self.mean) / self.std
    
    def inverse_transform(self, scaled):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted yet! Call fit or fit_transform first.")
        return (scaled * self.std) + self.mean

class MinMaxScaler:
    def __init__(self):
        self.max = None
        self.min = None
    
    def fit(self, X):
        self.max = np.max(X, axis=0)
        self.min = np.min(X, axis=0)
        
    def transform(self, X):
        if self.max is None or self.min is None:
            raise ValueError("Scaler not fitted yet! Call fit or fit_transform first.")
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X):
        self.max = np.max(X, axis=0)
        self.min = np.min(X, axis=0)
        
        return (X - self.min) / (self.max - self.min)
    
    def inverse_transform(self, scaled):
        if self.max is None or self.min is None:
            raise ValueError("Scaler not fitted yet! Call fit or fit_transform first.")
        return scaled * (self.max - self.min) + self.min

class OneHotEncoder:
    def __init__(self):
        self.categories = None

    def fit(self, X):
        self.categories = np.sort(np.unique(X))
    
    def transform(self, X):
        if self.categories is None:
            raise ValueError("Not Fitted yet")

        result = np.zeros((len(X), len(self.categories)))
        for i, cat in enumerate(self.categories):
            result[:, i] = (X == cat).astype(int)  # 1 where match, 0 else
        return result
    
    def fit_transform(self, X):
        self.categories = np.sort(np.unique(X))
        result = np.zeros((len(X), len(self.categories)))
        for i, cat in enumerate(self.categories):
            result[:, i] = (X == cat).astype(int)  # 1 where match, 0 else
        return result

    def get_features_names(self):
        if self.categories is None:
            raise ValueError("Not Fitted yet")
        return [f"is_{cat}" for cat in self.categories]

class LabelEncoder:
    def __init__(self):
        self.classes = None
    
    def fit(self, X):
        self.classes = np.sort(np.unique(X))
    
    def transform(self, X):
        if self.classes is None:
            raise ValueError("Not Fitted yet")
        invalid = ~np.isin(X, self.classes)
        if np.any(invalid):
            raise ValueError(f"Unknown categories: {np.unique(X[invalid])}")
        return np.searchsorted(self.classes, X)
    
    def fit_transform(self, X):
        self.classes = np.sort(np.unique(X))
        return np.searchsorted(self.classes, X)
    
    def inverse_transform(self, num):
        if self.classes is None:
            raise ValueError("Not Fitted yet")
        return self.classes[num]

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    """
    ...
    """
    def __init__(self, n_iter=200, learning_rate=0.01):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_pred = sigmoid((X @ self.weights) + self.bias)
            
            # calculate gradient
            dW = (1 / n_samples) * (X.T @ (y_pred - y))
            dB = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= dW * self.lr
            self.bias -= dB * self.lr
            
    def predict_proba(self, X):
        return sigmoid((X @ self.weights) + self.bias)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)