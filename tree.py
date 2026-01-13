import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth, min_samples_split, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
    # helper functions
    def _get_gini(self, y):
        if len(y) == 0:
            return 0
        classes, count = np.unique(y, return_counts=True)
        proba =  count / len(y)
        return 1 - np.sum(proba ** 2)
    
    def _get_entropy(self, y):
        # y: array-like of class labels
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    def _build_tree(self, X, y, depth=0):
        if ((len(np.unique(y)) == 1) or
        ((self.max_depth is not None) and (depth >= self.max_depth)) or
        (len(y) <= self.min_samples_split)):
            return Node(value=np.bincount(y).argmax())
            
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.bincount(y).argmax())
        
        # split samples
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def _best_split(self, X, y):
        best_score = np.inf
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            values = np.sort(np.unique(X[:, feature]))
            
            if len(values) < 2: # if length is 1 then skip
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                if self.criterion == "gini":
                    left_score = self._get_gini(y[left_idx])
                    right_score = self._get_gini(y[right_idx])
                elif self.criterion == "entropy":
                    left_score = self._get_entropy(y[left_idx])
                    right_score = self._get_entropy(y[right_idx])
                else: raise ValueError("Not a familiar criterion")    
                n_left = len(y[left_idx])
                n_right = len(y[right_idx])
                weighted_score = (n_left / n_samples) * left_score + (n_right / n_samples) * right_score
                if weighted_score < best_score:
                    best_score = weighted_score
                    best_threshold = threshold
                    best_feature = feature
        
        return best_feature, best_threshold    
    
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def fit(self, X, y):
       self.root = self._build_tree(X, y, depth=0) 
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def predict_proba(self, X):
        pass
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=None, criterion="gini"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.trees = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for n in range(self.n_estimators):
            rand_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boots_x = X[rand_idx]
            boots_y = y[rand_idx]
            
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_split, criterion=self.criterion)
            tree.fit(boots_x, boots_y)
            
            self.trees.append(tree)
    
    def predict(self, X):
        if len(self.trees) <= 1:
            raise ValueError("Not fitted yet!")
        trees_pred = np.array([tree.predict(X) for tree in self.trees]).T
        
        predictions = []
        for votes in trees_pred:
            pred = np.bincount(votes).argmax()
            predictions.append(pred)
        return np.array(predictions)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()