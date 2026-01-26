import numpy as np

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

class SimpleImputer:
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.st = None
    
    def fit(self, X):
        if self.strategy == "mean":
            self.st = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.st = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            X_cp = np.array([x for x in X if not self._is_missing(x)])
            values, counts = np.unique(X_cp, return_counts=True)
            self.st = values[np.argmax(counts)]

    def _is_missing(self, x):
        return (isinstance(x, float) and x != x) or (isinstance(x, str) and x.lower() == "nan")

    def transform(self, X):
        if self.st is None:
            raise ValueError("Not Fitted!")
        X = np.array([self.st if self._is_missing(x) else x for x in X])
        return X
    
    def fit_transform(self, X):
        self.fit(X) # fit on X
        return self.transform(X)