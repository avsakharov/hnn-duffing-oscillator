import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.notebook import tqdm


def setup_plot_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "text.usetex": True
    })


def make_phase_grid(system, grid_size):
    q_values = np.linspace(*system.q_range, grid_size)
    p_values = np.linspace(*system.p_range, grid_size)
    Q, P = np.meshgrid(q_values, p_values)
    x_values = np.stack([Q.flatten(), P.flatten()], axis=1)
    return Q, P, x_values


def plot_trajectory_and_phase_trajectory(t, trajectory, method_name="Numerical"):
    plt.figure(figsize=(10, 3.5))

    plt.subplot(1, 2, 1)
    plt.plot(t, trajectory[0], label='$q(t)$', color='tab:blue')
    plt.plot(t, trajectory[1], label='$p(t)$', color='tab:orange')
    plt.ylabel("$q$, $p$")
    plt.xlabel("$t$")
    plt.title(f"Coordinate and Momentum ({method_name})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trajectory[0], trajectory[1], color='tab:green')
    plt.xlabel("$q$")
    plt.ylabel("$p$")
    plt.title(f"Phase Trajectory ({method_name})")

    plt.tight_layout()
    plt.show()


def plot_phase_portraits(system, bnn, hnn, grid_size=30):
    Q, P, x_values = make_phase_grid(system, grid_size)

    dx_num_values = np.array([system.ode(0, x) for x in x_values])

    x_values = torch.tensor(x_values, dtype=torch.float32)
    with torch.no_grad():
        bnn.eval()
        dx_bnn_values = bnn.predict_derivatives(x_values).numpy()

        hnn.eval()
        dx_hnn_values = hnn.predict_derivatives(x_values).numpy()  # torch.enable_grad()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    titles = ("Numerical", "BaselineNN", "HamiltonianNN")
    fields = (dx_num_values, dx_bnn_values, dx_hnn_values)

    for i, ax in enumerate(axes):
        U = fields[i][:, 0].reshape(Q.shape)
        V = fields[i][:, 1].reshape(P.shape)
        ax.streamplot(Q, P, U, V, linewidth=0.5, density=1.5, arrowsize=0.5, color='tab:blue')
        ax.set_title(titles[i])
        ax.set_xlim(system.q_range)
        ax.set_ylim(system.p_range)
        ax.set_xlabel("$q$")
        ax.set_ylabel("$p$")

    fig.suptitle("Phase Portraits", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_hamiltonian_time_evolution(t, H_num, H_bnn, H_hnn):
    plt.figure(figsize=(10, 3.5))
    plt.plot(t, H_num, label='Numerical', linewidth=1, color='tab:brown')
    plt.plot(t, H_bnn, label='BaselineNN', linewidth=1, color='tab:green')
    plt.plot(t, H_hnn, label='HamiltonianNN', linewidth=1, color='tab:blue')
    plt.xlabel("$t$")
    plt.ylabel("$H$")
    plt.title("Hamiltonian Evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hamiltonian_surfaces_comparison(system, hnn, grid_size=100):
    Q, P, x_values = make_phase_grid(system, grid_size)

    H_true = np.array([system.get_hamiltonian(x) for x in x_values]).reshape(grid_size, grid_size)

    with torch.no_grad():
        hnn.eval()
        x_values = torch.tensor(x_values, dtype=torch.float32)
        H_hnn = hnn.forward(x_values).squeeze().numpy().reshape(grid_size, grid_size)

    Z = (
        (H_true, "True Hamiltonian Surface"),
        (H_hnn, "Predicted Hamiltonian Surface (HamiltonianNN)"),
        (np.abs(H_true - H_hnn), "Absolute Error")
    )

    fig = plt.figure(figsize=(10, 3.5))

    for i, (z, title) in enumerate(Z, start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.plot_surface(Q, P, z, cmap=cm.viridis)
        ax.set_xlabel("$q$")
        ax.set_ylabel("$p$")
        ax.view_init(elev=30, azim=110)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_phase_trajectories(system, bnn, hnn, x0_values):
    methods = (
        ("Numerical", lambda: system.simulate_trajectories_numerical(x0_values)),
        ("BaselineNN", lambda: system.simulate_trajectories_by_model(bnn, x0_values)),
        ("HamiltonianNN", lambda: system.simulate_trajectories_by_model(hnn, x0_values)),
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, (title, simulate_func) in zip(axes, methods):
        ax.set_title(title)
        trajectories = simulate_func()  # (N, 2, T)

        for trajectory in trajectories:
            ax.plot(trajectory[0], trajectory[1], linewidth=0.5, color='tab:blue')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-5.5, 5.5)
        ax.set_xlabel("$q$")
        ax.set_ylabel("$p$")

    fig.suptitle("Phase Trajectories", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_poincare_sections(system, bnn, hnn, x0_values, n_points=200, seed=42):
    n_colors = len(x0_values)
    cmap = plt.colormaps["Dark2"]
    colors = [cmap(i / n_colors) for i in range(n_colors)]
    random.seed(seed)
    random.shuffle(colors)

    methods = (
        ("Numerical", lambda x0_values: system.simulate_trajectories_numerical(x0_values)),
        ("BaselineNN", lambda x0_values: system.simulate_trajectories_by_model(bnn, x0_values)),
        ("HamiltonianNN", lambda x0_values: system.simulate_trajectories_by_model(hnn, x0_values))
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, (title, simulate_func) in zip(axes, methods):
        x0_values_ = x0_values.copy()

        for _ in tqdm(range(n_points), desc=f"Computing Poincaré section ({title})"):
            trajectories = simulate_func(x0_values_)  # (N, 2, T)
            x0_values_ = trajectories[:, :, -1]  # (N, 2)

            for i, (u, v) in enumerate(x0_values_):
                ax.scatter(u, v, s=0.5, color=colors[i])

        ax.set_title(f"{title}")
        ax.set_xlabel("$u$")
        ax.set_ylabel("$v$")

    fig.suptitle(rf"Poincaré Sections for $\gamma={system.gamma}$, $\varepsilon={system.epsilon}$", fontsize=14)
    plt.tight_layout()
    plt.show()
