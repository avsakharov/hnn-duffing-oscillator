from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.integrate import solve_ivp
import numpy as np
import torch

torch.manual_seed(42)


class DuffingSystem:
    """
    Base class for Duffing oscillators.
    """

    def __init__(self, t_span, dt, seed):
        self.t_span = t_span
        self.dt = dt
        self.seed = seed

    @property
    def t_eval(self):
        return np.arange(*self.t_span, self.dt)

    def simulate_trajectory_numerical(self, x0):
        sol = solve_ivp(
            self.ode,
            self.t_span,
            x0,
            t_eval=self.t_eval,
            rtol=1e-9,
            atol=1e-9
        )
        return sol.y  # trajectory (2, T)


class ConservativeDuffingSystem(DuffingSystem):
    """
    Conservative Duffing oscillator.
    """

    def __init__(self, t_span=(0, 20), dt=0.01, seed=42, q_range=(-2, 2), p_range=(-2, 2)):
        super().__init__(t_span, dt, seed)
        self.q_range = q_range
        self.p_range = p_range

    @staticmethod
    def get_hamiltonian(x):
        """
        H(q, p) = 0.5 * p^2 - 0.5 * q^2 + 0.25 * q^4
        """
        q, p = x
        return 0.5 * p**2 - 0.5 * q**2 + 0.25 * q**4

    @staticmethod
    def ode(t, x):
        """
        dq/dt = p,
        dp/dt = q - q^3
        """
        q, p = x
        return np.array((p, q - q**3))

    def simulate_trajectories_numerical(self, x0_values):
        trajectories = []
        for x0 in x0_values:
            trajectory = self.simulate_trajectory_numerical(x0)
            trajectories.append(trajectory)
        return np.stack(trajectories, axis=0)  # trajectories (N, 2, T)

    def simulate_trajectory_by_model(self, model, x0):
        """
        Integrate using a neural network model and 4th-order Runge-Kutta.
        """
        x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
        trajectory = []

        with torch.no_grad():
            f = model.predict_derivatives

            for _ in self.t_eval:
                trajectory.append(x.numpy()[0])

                k1 = f(x)
                k2 = f(x + 0.5 * self.dt * k1)
                k3 = f(x + 0.5 * self.dt * k2)
                k4 = f(x + self.dt * k3)
                x = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return np.array(trajectory).T  # trajectory (2, T)

    def simulate_trajectories_by_model(self, model, x0_values):
        x_values = torch.tensor(x0_values, dtype=torch.float32)  # (N, 2)
        N = x_values.shape[0]
        T = len(self.t_eval)
        trajectories = torch.zeros((T, N, 2), dtype=torch.float32)

        with torch.no_grad():
            f = model.predict_derivatives

            for i, _ in enumerate(self.t_eval):
                trajectories[i] = x_values
                k1 = f(x_values)
                k2 = f(x_values + 0.5 * self.dt * k1)
                k3 = f(x_values + 0.5 * self.dt * k2)
                k4 = f(x_values + self.dt * k3)
                x_values = x_values + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return trajectories.permute(1, 2, 0).numpy()  # trajectories (N, 2, T)

    def generate_dataset(self, n_samples=100_000):
        """
        Generate training and test data by sampling phase space.
        """
        np.random.seed(self.seed)
        q_values = np.random.uniform(*self.q_range, size=n_samples)
        p_values = np.random.uniform(*self.p_range, size=n_samples)

        X = np.column_stack((q_values, p_values))  # (N, 2)
        y = np.array([self.ode(0, x) for x in X])  # (N, 2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=1024, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.float32)),
            batch_size=1024, shuffle=False
        )

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
        return train_loader, test_loader


class ChaoticDuffingSystem(DuffingSystem):
    """
    Duffing oscillator with dissipation and external periodic force.
    """

    def __init__(self, t_span=(0, 2*np.pi), dt=0.01, seed=42,
                 u_range=(-2, 2), v_range=(-2, 2), gamma=0.0, epsilon=0.0):
        super().__init__(t_span, dt, seed)
        self.u_range = u_range
        self.v_range = v_range
        self.gamma = gamma
        self.epsilon = epsilon

    def ode(self, t, x):
        """
        du/dt = v
        dv/dt = -gamma * v + u - u**3 + epsilon * sin(t)
        """
        u, v = x
        g = self.gamma
        e = self.epsilon
        return np.array((v, -g * v + u - u ** 3 + e * np.sin(t)))

    def simulate_trajectories_numerical(self, x0_values):
        trajectories = []

        for x0 in x0_values:
            trajectory = self.simulate_trajectory_numerical(x0)
            trajectories.append(trajectory)

        return np.stack(trajectories, axis=0)  # (N, 2, T)

    def simulate_trajectories_by_model(self, model, x0_values):
        T = len(self.t_eval)
        x_values = torch.tensor(x0_values, dtype=torch.float32)  # (N, 2)
        N = x_values.shape[0]
        trajectories = torch.zeros((T, N, 2), dtype=torch.float32)

        sin_cos = torch.tensor(np.stack([
            np.sin(self.t_eval), np.cos(self.t_eval),
            np.sin(self.t_eval + 0.5 * self.dt), np.cos(self.t_eval + 0.5 * self.dt),
            np.sin(self.t_eval + self.dt), np.cos(self.t_eval + self.dt)
        ], axis=1), dtype=torch.float32)  # (T, 6)

        with torch.no_grad():
            f = model.predict_derivatives

            for i in range(T):
                trajectories[i] = x_values

                sin_cos1 = sin_cos[i, 0:2].repeat(N, 1)
                sin_cos2 = sin_cos[i, 2:4].repeat(N, 1)  # sin_cos3 = sin_cos2
                sin_cos4 = sin_cos[i, 4:6].repeat(N, 1)

                k1 = f(torch.cat([x_values, sin_cos1], dim=1))
                k2 = f(torch.cat([x_values + 0.5 * self.dt * k1, sin_cos2], dim=1))
                k3 = f(torch.cat([x_values + 0.5 * self.dt * k2, sin_cos2], dim=1))  # sin_cos3 = sin_cos2
                k4 = f(torch.cat([x_values + self.dt * k3, sin_cos4], dim=1))

                x_values = x_values + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return trajectories.permute(1, 2, 0).numpy()  # (N, 2, T)

    def generate_dataset(self, n_samples=100_000):
        """
        Generate training and test data by sampling extended phase space.
        """
        np.random.seed(self.seed)
        u_values = np.random.uniform(*self.u_range, size=n_samples)
        v_values = np.random.uniform(*self.v_range, size=n_samples)
        t_values = np.random.uniform(0, 2 * np.pi, size=n_samples)

        X = np.column_stack((u_values, v_values, np.sin(t_values), np.cos(t_values)))  # (N, 4)
        y = np.array([self.ode(t, (u, v)) for u, v, t in zip(u_values, v_values, t_values)])  # (N, 2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=1024, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.float32)),
            batch_size=1024, shuffle=False
        )

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
        return train_loader, test_loader
