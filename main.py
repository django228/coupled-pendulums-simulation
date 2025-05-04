import configparser
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Загрузка конфигурации
config = configparser.ConfigParser()
config.read("config.ini")

# Начальные условия
theta1 = float(config["InitialConditions"]["theta1"])  # начальный угол первого маятника (в радианах)
omega1 = float(config["InitialConditions"]["omega1"])  # начальная угловая скорость первого маятника
theta2 = float(config["InitialConditions"]["theta2"])  # начальный угол второго маятника
omega2 = float(config["InitialConditions"]["omega2"])  # начальная угловая скорость второго маятника
y0 = [theta1, omega1, theta2, omega2]

# Параметры маятников
L = float(config["Pendulums"]["L"])  # Длина маятников (м)
m = float(config["Pendulums"]["m"])  # Масса (кг)
g = float(config["Pendulums"]["g"])  # Ускорение свободного падения (м/с²)
k = float(config["Pendulums"]["k"])  # Жесткость пружины (Н/м)
L1 = float(config["Pendulums"]["L1"])  # Расстояние до точки крепления пружины (м)
beta = float(config["Pendulums"]["beta"])  # Коэффицент затухания

# --- Уравнения движения ---
def equations(t, y):
    θ1, ω1, θ2, ω2 = y
    dθ1 = ω1
    dθ2 = ω2
    dω1 = (-beta*ω1 - m*g*θ1 - k*L1**2*(θ1 - θ2)) / (m*L**2)
    dω2 = (-beta*ω2 - m*g*θ2 - k*L1**2*(θ2 - θ1)) / (m*L**2)
    return [dθ1, dω1, dθ2, dω2]

t_eval = np.linspace(0, 40, 4000)
sol = solve_ivp(equations, [0, 40], y0, t_eval=t_eval)
t, θ1, ω1, θ2, ω2 = sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(t, θ1, label="θ₁ (Маятник 1)")
plt.plot(t, θ2, label="θ₂ (Маятник 2)", linestyle='--')
plt.title("Углы отклонения во времени")
plt.xlabel("Время (с)")
plt.ylabel("Угол (рад)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "углы.png"))
plt.close()

# 2. График скоростей
plt.figure(figsize=(10, 5))
plt.plot(t, ω1, label="ω₁ (Маятник 1)")
plt.plot(t, ω2, label="ω₂ (Маятник 2)", linestyle='--')
plt.title("Угловые скорости во времени")
plt.xlabel("Время (с)")
plt.ylabel("Угловая скорость (рад/с)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "скорости.png"))
plt.close()

# 3. Два графика вместе
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, θ1, label="θ₁ (Маятник 1)")
plt.plot(t, θ2, label="θ₂ (Маятник 2)", linestyle='--')
plt.title("Углы отклонения связанных маятников")
plt.ylabel("Угол (рад)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, ω1, label="ω₁ (Маятник 1)")
plt.plot(t, ω2, label="ω₂ (Маятник 2)", linestyle='--')
plt.title("Угловые скорости связанных маятников")
plt.xlabel("Время (с)")
plt.ylabel("Угл. скорость (рад/с)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "общий_график.png"))
plt.close()

print("Графики сохранены в папке: output/")