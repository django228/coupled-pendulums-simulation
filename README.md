# Coupled Pendulums Simulation

Simulation of two coupled pendulums connected by a spring using Python.  
This project numerically solves and visualizes the motion of a damped two-pendulum system with spring coupling.

---

## Features

- Models the dynamics of two identical pendulums with:
  - Spring coupling at a fixed distance from suspension point
  - Linear damping
  - Small angle approximation
- Solves equations using `scipy.integrate.solve_ivp`
- Plots angular displacement over time

---

## Mathematical Model

The equations of motion for small angles:

\[
mL^2 \ddot{\theta}_1 + \beta L^2 \dot{\theta}_1 + mgL \theta_1 + k L_1^2 (\theta_1 - \theta_2) = 0
\]
\[
mL^2 \ddot{\theta}_2 + \beta L^2 \dot{\theta}_2 + mgL \theta_2 + k L_1^2 (\theta_2 - \theta_1) = 0
\]

---

## Getting Started

### Requirements

- Python 3.8+
- `numpy`
- `scipy`
- `matplotlib`

### Run

Clone the repository:

```bash
git clone https://github.com/yourusername/coupled-pendulums-simulation.git
cd coupled-pendulums-simulation
