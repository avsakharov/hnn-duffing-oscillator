import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class BaseDynamicsNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.train_loss_history = []
        self.test_loss_history = []

    def loss(self, x, dx_true):
        dx_pred = self.predict_derivatives(x)
        return ((dx_pred - dx_true) ** 2).mean()

    def train_model(self, train_loader, test_loader, optimizer, epochs=10):
        for epoch in tqdm(range(epochs), desc=f"Training {self.__class__.__name__}"):
            self.train()
            total_train_loss = 0.0
            for xb, yb in train_loader:
                loss = self.loss(xb, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            self.eval()
            total_test_loss = sum(self.loss(xb, yb).item() for xb, yb in test_loader)

            self.train_loss_history.append(total_train_loss / len(train_loader))
            self.test_loss_history.append(total_test_loss / len(test_loader))

            print(f"Epoch {epoch+1:02d}: train_loss = {self.train_loss_history[-1]:.6f}, "
                  f"test_loss = {self.test_loss_history[-1]:.6f}")

    def plot_loss(self):
        plt.figure(figsize=(4.5, 3))
        plt.plot(self.train_loss_history, label='Train Loss', color='tab:blue')
        plt.plot(self.test_loss_history, label='Test Loss', color='tab:orange')
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title(f"{self.__class__.__name__} Training and Test Loss")
        plt.legend()
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.show()

    def predict_derivatives(self, x):
        raise NotImplementedError


class BaselineNN(BaseDynamicsNN):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2):
        super().__init__(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.net(x)

    def predict_derivatives(self, x):
        return self.forward(x)


class HamiltonianNN(BaseDynamicsNN):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        super().__init__(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.net(x)

    def predict_derivatives(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            H = self.forward(x)
            grad = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            return torch.stack([grad[:, 1], -grad[:, 0]], dim=1)
