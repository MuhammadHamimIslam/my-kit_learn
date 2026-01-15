import numpy as np

class NeuralNet:
    def __init__(self, epochs=300, learning_rate=0.05, batch_size=32):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def _forward_pass(self, x):
        return x @ self.weights + self.bias

    def _gradient(self, x, y, y_pred):
        error = y_pred - y
        dW = x.T @ error / x.shape[0]   # shape: (1, n_features)
        dB = np.mean(error)             # scalar
        return dW, dB
    
    def _loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, X, y, verbose=1):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))   # ← better init than random
        self.bias = 0.0
        losses = []
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuf = X[indices]
            y_shuf = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                x_batch = X_shuf[i:i+self.batch_size]
                y_batch = y_shuf[i:i+self.batch_size]
                
                y_pred = self._forward_pass(x_batch)
                dW, dB = self._gradient(x_batch, y_batch, y_pred)
                
                self.weights -= self.lr * dW
                self.bias    -= self.lr * dB
            
            y_pred_full = self._forward_pass(X)
            loss = self._loss(y, y_pred_full)
            losses.append(loss)
            
            if verbose and (epoch % 50 == 0 or epoch == self.epochs-1):
                print(f"Epoch {epoch:3d} | MSE = {loss:.7f} | w = {self.weights[0,0]:.5f} | b = {self.bias:.5f}")
        
        return losses
    
    def predict(self, X):
        return self._forward_pass(X)