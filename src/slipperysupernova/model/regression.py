import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Model parameters
        self.w_1 = nn.Parameter(torch.randn(1, requires_grad=True))  # slope
        self.w_0 = nn.Parameter(torch.randn(1, requires_grad=True))  # intercept
        
        # Training data storage
        self.X_train = None
        self.y_train = None

        # Testing data Storage
        self.X_test = None
        self.y_test = None

        self.fitted = False
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD([self.w_1, self.w_0], lr=self.learning_rate)
        
        # Training history
        self.loss_history = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.
        
        Args:
            X: Input tensor of shape (n_samples,)
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        return self.w_1 * X + self.w_0
     
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,X_test: np.ndarray, y_test: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data.
        
        Args:
            X: Input features of shape (n_samples,)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Returns the fitted model instance
        """
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        
        # Training loop
        prev_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.forward(self.X_train)
            
            # Compute loss
            loss = self.criterion(y_pred, self.y_train)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Store loss history
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            
            prev_loss = current_loss
        
        #R^2
        with torch.no_grad():
            test_preds = self.forward(self.X_test)
            ss_res = torch.sum((self.y_test - test_preds) ** 2)
            ss_tot = torch.sum((self.y_test - torch.mean(self.y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot).item() if ss_tot.item() != 0 else 0.0
        self.fitted = True
        print(f"Training complete. R^2 Score on Test Data: {r2:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples,)
            
        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.numpy()

    def analysis_plot(self):
        """
        Creates subplots on a single figure displaying original data, 
        fitted regression line, and the progression of w0, w1, and loss.
        """    
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Original data and fitted regression line
        axs[0, 0].scatter(self.X_train, self.y_train, color='blue', label='Train Data', alpha=0.6)
        axs[0, 0].scatter(self.X_test, self.y_test, color='orange', label='Test Data', alpha=0.6)
        x_line = np.linspace(min(min(self.X_train), min(self.X_test)), max(max(self.X_train), max(self.X_test)), 100)
        y_line = self.predict(x_line)
        axs[0, 0].plot(x_line, y_line, color='red', label='Fitted Line')
        axs[0, 0].set_title('Data & Fitted Regression Line')
        axs[0, 0].set_xlabel('X')
        axs[0, 0].set_ylabel('Y')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # 2. Loss progression
        axs[0, 1].plot(self.history_loss, color='purple')
        axs[0, 1].set_title('MSE Loss Progression')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
        
        # 3. w0 (bias) progression
        axs[1, 0].plot(self.history_w0, color='green')
        axs[1, 0].set_title('Bias (w0) Progression')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('w0')
        axs[1, 0].grid(True)
        
        # 4. w1 (weight) progression
        axs[1, 1].plot(self.history_w1, color='brown')
        axs[1, 1].set_title('Weight (w1) Progression')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('w1')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()