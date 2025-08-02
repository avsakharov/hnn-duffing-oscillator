# Hamiltonian Neural Network for the Duffing Oscillator

This project explores the use of **Hamiltonian Neural Network (HNN)** to learn and simulate the dynamics of the **Duffing oscillator** and compares their performance to baseline neural networks.

## Overview

The Duffing oscillator is a nonlinear second-order system. The goal of this project is to:

- Learn the vector field of the system using two types of neural networks:
  - **BaselineNN**: Predicts derivatives directly.
  - **HamiltonianNN**: Learns the Hamiltonian and derives equations via automatic differentiation.
- Compare the learned dynamics with numerical simulations (ground truth).

## System Variants

### Conservative Duffing Oscillator

Hamiltonian form:

```math
H(q, p) = \frac{p^2}{2} - \frac{q^2}{2} + \frac{q^4}{4}.
```

Equations of motion:

```math
\dot{q} = \frac{\partial H}{\partial p} = p, \quad
\dot{p} = -\frac{\partial H}{\partial q} = q - q^3.
```

Here, $q$ and $p$ are the canonical coordinate and momentum.

### Chaotic Duffing Oscillator

This version includes damping and a periodic external force:

```math
\dot{u} = v, \quad
\dot{v} = -\gamma v + u - u^3 + \varepsilon \sin t,
```

where $u$ and $v$ represent the coordinate and velocity, respectively, $\gamma$ is the damping coefficient, $\varepsilon$ is the amplitude of the external periodic force.

## Installation

Clone the repository:

```bash
git clone https://github.com/avsakharov/hnn-duffing-oscillator.git
cd hnn-duffing-oscillator
```

## Libraries used

numpy, torch, matplotlib, tqdm, scipy, scikit-learn, random, os, math
