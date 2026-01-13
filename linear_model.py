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