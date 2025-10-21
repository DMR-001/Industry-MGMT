import numpy as np

class LinearRegression:
    """Linear Regression implementation from scratch using gradient descent"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """Train the linear regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate cost
            cost = self.calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return np.dot(X, self.weights) + self.bias
    
    def calculate_cost(self, y_true, y_pred):
        """Calculate mean squared error cost"""
        n_samples = len(y_true)
        cost = (1/(2*n_samples)) * np.sum((y_pred - y_true)**2)
        return cost
    
    def rmse(self, y_true, y_pred):
        """Calculate Root Mean Square Error"""
        mse = np.mean((y_true - y_pred)**2)
        return np.sqrt(mse)
    
    def mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def r2_score(self, y_true, y_pred):
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def get_parameters(self):
        """Get model parameters"""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'cost_history': self.cost_history
        }