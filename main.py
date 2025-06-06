import configparser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib;

matplotlib.use("TkAgg")

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C']

config = configparser.ConfigParser()
config.read("config.ini")

# Начальные условия
theta1_0 = float(config["InitialConditions"]["theta1"])  # начальный угол первого маятника (в радианах)
omega1_0 = float(config["InitialConditions"]["omega1"])  # начальная угловая скорость первого маятника
theta2_0 = float(config["InitialConditions"]["theta2"])  # начальный угол второго маятника
omega2_0 = float(config["InitialConditions"]["omega2"])  # начальная угловая скорость второго маятника

# Параметры маятников
L = float(config["Pendulums"]["L"])  # Длина маятников (м)
m = float(config["Pendulums"]["m"])  # Масса (кг)
g = float(config["Pendulums"]["g"])  # Ускорение свободного падения (м/с²)
k = float(config["Pendulums"]["k"])  # Жесткость пружины (Н/м)
L1 = float(config["Pendulums"]["L1"])  # Расстояние до точки крепления пружины (м)
beta = float(config["Pendulums"]["beta"])  # Коэффицент затухания
r = float(config["Pendulums"]["r"])  # Радиус шариков(м)

# Временной интервал
t_start = float(config["Settings.TimeInterval"]["t_start"])
t_end = float(config["Settings.TimeInterval"]["t_end"])

# Уравнения движения с учетом пружины между маятниками
def equations_with_spring(t, y):
    theta1, omega1, theta2, omega2 = y

    # Позиции маятников
    x1 = L * np.sin(theta1) - L1 / 2
    y1 = -L * np.cos(theta1)
    x2 = L * np.sin(theta2) + L1 / 2
    y2 = -L * np.cos(theta2)

    # Расстояние между маятниками
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    F_spring = k * dist

    F1_theta = -F_spring * (x2 - x1) * L * np.cos(theta1) - F_spring * (y2 - y1) * L * np.sin(theta1)
    F2_theta = F_spring * (x2 - x1) * L * np.cos(theta2) + F_spring * (y2 - y1) * L * np.sin(theta2)

    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt = -(beta / m) * omega1 - (g / L) * np.sin(theta1) + F1_theta / (m * L ** 2)
    domega2_dt = -(beta / m) * omega2 - (g / L) * np.sin(theta2) + F2_theta / (m * L ** 2)

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]


# Событие столкновения
def collision_event(t, y):
    theta1, omega1, theta2, omega2 = y
    x1 = L * np.sin(theta1) - L1 / 2
    y1 = -L * np.cos(theta1)
    x2 = L * np.sin(theta2) + L1 / 2
    y2 = -L * np.cos(theta2)
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist - 2 * r  # Расстояние между центрами минус диаметр


collision_event.terminal = True
collision_event.direction = -1


# Обработка столкновений
def handle_collision(y):
    theta1, omega1, theta2, omega2 = y
    v1 = omega1 * L
    v2 = omega2 * L
    v1_new = v2  # Для одинаковых масс просто обмен скоростями
    v2_new = v1
    return [theta1, v1_new / L, theta2, v2_new / L]


# Параметры интегрирования
sol_params = {
    'method': 'RK45',
    'atol': 1e-8,
    'rtol': 1e-6,
    'max_step': 0.01
}

# Интегрирование
t_current = t_start
y_current = [theta1_0, omega1_0, theta2_0, omega2_0]
t_list = []
y_list = []

while t_current < t_end:
    sol = solve_ivp(
        fun=equations_with_spring,
        t_span=(t_current, t_end),
        y0=y_current,
        events=collision_event,
        **sol_params
    )
    t_list.extend(sol.t)
    y_list.append(sol.y)

    if sol.status == 1:  # Произошло событие
        t_col = sol.t_events[0][0]
        y_col = sol.y_events[0][0]
        theta1, omega1, theta2, omega2 = y_col
        x1 = L * np.sin(theta1) - L1 / 2
        y1 = -L * np.cos(theta1)
        x2 = L * np.sin(theta2) + L1 / 2
        y2 = -L * np.cos(theta2)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist <= 2 * r * 1.05:
            y_current = handle_collision(y_col)
            t_current = t_col
        else:
            break
    else:
        break

# Обработка результатов
t_all = np.array(t_list)
y_all = np.hstack(y_list)
theta1, omega1, theta2, omega2 = y_all

# Графики углов и скоростей
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# График углов
ax1.plot(t_all, theta1, label=r'$\theta_1(t)$', color=colors[0], linewidth=2)
ax1.plot(t_all, theta2, label=r'$\theta_2(t)$', color=colors[1], linewidth=2)
ax1.set_ylabel('Угол (рад)', fontsize=12)
ax1.legend(fontsize=12, loc='upper right')
ax1.set_title('Углы маятников', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3)

# График угловых скоростей
ax2.plot(t_all, omega1, label=r'$\omega_1(t)$', color=colors[0], linewidth=2)
ax2.plot(t_all, omega2, label=r'$\omega_2(t)$', color=colors[1], linewidth=2)
ax2.set_xlabel('Время, с', fontsize=12)
ax2.set_ylabel('Угловая скорость (рад/с)', fontsize=12)
ax2.legend(fontsize=12, loc='upper right')
ax2.set_title('Угловые скорости', fontsize=14, pad=20)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анимация
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1.5 * L, 1.5 * L)
ax.set_ylim(-1.5 * L, 1.5 * L)
ax.set_aspect('equal')
ax.set_facecolor('#F5F5F5')
ax.grid(True, alpha=0.3)
ax.set_title('Анимация связанных маятников', fontsize=14, pad=20)

# Элементы анимации
line1, = ax.plot([], [], 'o-', lw=3, color=colors[0], markersize=r * 120,
                 markerfacecolor=colors[0], markeredgecolor='black', label='Маятник 1')
line2, = ax.plot([], [], 'o-', lw=3, color=colors[1], markersize=r * 120,
                 markerfacecolor=colors[1], markeredgecolor='black', label='Маятник 2')
connection, = ax.plot([], [], 'k-', lw=2, label='Ось')
spring, = ax.plot([], [], color=colors[3], lw=3, alpha=0.7, label='Пружина')
ax.legend(fontsize=12, loc='upper right')


def init_anim():
    line1.set_data([], [])
    line2.set_data([], [])
    connection.set_data([], [])
    spring.set_data([], [])
    return line1, line2, connection, spring


def animate_frame(i):
    th1, th2 = theta1[i], theta2[i]
    x1 = L * np.sin(th1) - L1 / 2
    y1 = -L * np.cos(th1)
    x2 = L * np.sin(th2) + L1 / 2
    y2 = -L * np.cos(th2)
    connection.set_data([-L1 / 2, L1 / 2], [0, 0])
    line1.set_data([-L1 / 2, x1], [0, y1])
    line2.set_data([L1 / 2, x2], [0, y2])

    # Создаем эффект пружины (зигзагообразная линия)
    n_spring = 15
    spring_x = np.linspace(x1, x2, n_spring)
    spring_y = np.linspace(y1, y2, n_spring)
    spring_amp = 0.05 * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    spring_y += spring_amp * np.sin(np.linspace(0, 5 * np.pi, n_spring))
    spring.set_data(spring_x, spring_y)

    return line1, line2, connection, spring


ani = FuncAnimation(fig, animate_frame, frames=len(t_all), init_func=init_anim,
                    blit=True, interval=20)
plt.show()

# Нормальные частоты и период биений
A = np.array([[-g / L + (k * L1) / (m * L), - (k * L1) / (m * L)],
              [- (k * L1) / (m * L), -g / L + (k * L1) / (m * L)]])
eigs = np.linalg.eigvals(A)
omega1_norm = np.sqrt(abs(eigs[0]))
omega2_norm = np.sqrt(abs(eigs[1]))
delta_omega = abs(omega2_norm - omega1_norm)
T_beat = 2 * np.pi / delta_omega

# Красивое отображение результатов
print("\n" + "=" * 50)
print(f"{'Нормальные частоты системы':^50}")
print("=" * 50)
print(f"ω₁ = {omega1_norm:.3f} рад/с")
print(f"ω₂ = {omega2_norm:.3f} рад/с")
print("-" * 50)
print(f"{'Период биений':^50}")
print(f"T ≈ {T_beat:.2f} с")
print("=" * 50 + "\n")