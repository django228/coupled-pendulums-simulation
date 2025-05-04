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

<pre> mL² θ̈₁ + βL² θ̇₁ + mgL θ₁ + kL₁²(θ₁ - θ₂) = 0 mL² θ̈₂ + βL² θ̇₂ + mgL θ₂ + kL₁²(θ₂ - θ₁) = 0 </pre>

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
