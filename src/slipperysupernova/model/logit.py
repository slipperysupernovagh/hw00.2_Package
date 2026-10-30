import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, lr=0.01, epochs=500):
        super(LogisticRegression, self).__init__()
        # Linear layer: weights and bias
        self.linear = nn.Linear(input_dim, 1)
        self.epochs = epochs
        self.lr = lr
        
    def forward(self, x):
        # Apply sigmoid activation to get probabilities between 0 and 1
        return torch.sigmoid(self.linear(x))
    
    def fit(self, X, y, batch_size=64):
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # Binary Cross Entropy Loss
        criterion = nn.BCELoss()
        # Stochastic Gradient Descent Optimizer
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            total_loss = epoch_loss / len(dataset)
            history.append(total_loss)
            
            if (epoch + 1) % (self.epochs // 5) == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss:.4f}")
                
        return history

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).numpy().flatten()
            
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)