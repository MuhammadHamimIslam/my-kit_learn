import numpy as np

class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        self.classes = None
        self.x_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def predict(self, X):
        if (self.x_train is None) or (self.y_train is None) or (self.classes is None):
            raise ValueError("Not Fitted yet")
        predictions = []
        for x in X:
            d = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
            k_idx = np.argsort(d)[:self.k]
            k_lables = self.y_train[k_idx]
            pred = np.bincount(k_lables).argmax()
            predictions.append(pred)
        return np.array(predictions)
    
    def predict_proba(self, X):
        if (self.x_train is None) or (self.y_train is None) or (self.classes is None):
            raise ValueError("Not Fitted yet")
        probas = []
        for x in X:
            d = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
            k_idx = np.argsort(d)[:self.k]
            k_lables = self.y_train[k_idx]
            
            proba = np.zeros(len(self.classes))
            for i, cls in enumerate(self.classes):
                proba[i] = np.sum(k_lables == cls) / self.k
            probas.append(proba)
        return np.array(probas)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

class KNeighborsRegressor:
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X, method="mean"):
        if (self.x_train is None) or (self.y_train is None):
            raise ValueError("Not Fitted yet")
        predictions = []
        for x in X:
            d = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
            k_idx = np.argsort(d)[:self.k]
            k_target = self.y_train[k_idx]
            
            if method == "mean":
                pred = np.mean(k_target)
                predictions.append(pred)
            elif method == "median":
                pred = np.median(k_target)
                predictions.append(pred)
            else:
                raise ValueError("Invalid method")

        return np.array(predictions)

    def score(self, X, y, metric="MSE"):
        """
        Args -> X: test data, y: label
        returns -> mean squared error, root mean squared error, R² score
        """
        y_pred = self.predict(X)
        error = y_pred - y
        if metric == "MSE":
            return np.mean(error ** 2)
        elif metric == "RMSE":
            return np.sqrt(np.mean(error ** 2))
        elif metric == "r2":
            return 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))
        else:
            raise ValueError("Not a familiar metric!")