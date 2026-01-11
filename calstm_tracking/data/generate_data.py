"""
generate_data.py
----------------
生成高机动空中目标的三维轨迹数据（真值 + 含噪测量）

状态定义：
    x = [x, y, z, vx, vy, vz]

输出：
    state_true:      (N, 6)
    measurement:     (N, 6)
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 基本参数设置
# =========================

DT = 0.5                 # 雷达采样周期（秒）
TOTAL_TIME = 100         # 总时长（秒）
N = int(TOTAL_TIME / DT)

G = 9.81                 # 重力加速度
G_MAX = 8                # 最大机动过载（8G，高机动）
A_MAX = G_MAX * G

# 噪声标准差（可调）
POS_NOISE_STD = 300.0    # 位置噪声（米）
VEL_NOISE_STD = 20.0     # 速度噪声（米/秒）


# =========================
# 2. 机动加速度模型
# =========================

def get_acceleration(t):
    """
    分段定义加速度，模拟高机动目标
    """
    if t < 20:
        return np.array([0.0, 0.0, 0.0])                     # 匀速
    elif t < 40:
        return np.array([0.3 * A_MAX, 0.0, 0.0])             # 水平机动
    elif t < 60:
        return np.array([0.0, 0.0, 0.5 * A_MAX])             # 爬升
    elif t < 80:
        return np.array([-0.4 * A_MAX, 0.2 * A_MAX, 0.0])    # 复合机动
    else:
        return np.array([0.0, 0.0, -0.4 * A_MAX])            # 俯冲


# =========================
# 3. 轨迹生成函数
# =========================

def generate_trajectory():
    """
    生成一条完整轨迹
    """
    state_true = np.zeros((N, 6))

    # 初始状态（典型空中目标）
    state_true[0, :] = np.array([
        0.0,            # x (m)
        0.0,            # y (m)
        10000.0,        # z (m)
        250.0,          # vx (m/s)
        0.0,            # vy (m/s)
        0.0             # vz (m/s)
    ])

    for k in range(1, N):
        t = k * DT
        a = get_acceleration(t)

        # 位置更新
        state_true[k, 0:3] = (
            state_true[k-1, 0:3]
            + state_true[k-1, 3:6] * DT
            + 0.5 * a * DT ** 2
        )

        # 速度更新
        state_true[k, 3:6] = (
            state_true[k-1, 3:6]
            + a * DT
        )

    return state_true


# =========================
# 4. 观测生成（加噪声）
# =========================

def generate_measurement(state_true):
    """
    生成含噪测量
    """
    measurement = state_true.copy()

    measurement[:, 0:3] += np.random.randn(N, 3) * POS_NOISE_STD
    measurement[:, 3:6] += np.random.randn(N, 3) * VEL_NOISE_STD

    return measurement


# =========================
# 5. 可视化（非常重要）
# =========================

def plot_trajectory(state_true, measurement):
    plt.figure(figsize=(10, 5))

    plt.plot(
        state_true[:, 0],
        state_true[:, 1],
        label="True Trajectory",
        linewidth=2
    )

    plt.scatter(
        measurement[:, 0],
        measurement[:, 1],
        s=10,
        alpha=0.4,
        label="Measured"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("High Maneuvering Target Trajectory (XY Plane)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()


# =========================
# 6. 主函数
# =========================

def main(save_path="trajectory.npz"):
    state_true = generate_trajectory()
    measurement = generate_measurement(state_true)

    # 保存数据
    np.savez(
        save_path,
        state_true=state_true,
        measurement=measurement
    )

    print("Data saved to:", save_path)
    print("State true shape:", state_true.shape)
    print("Measurement shape:", measurement.shape)

    # 画图检查
    plot_trajectory(state_true, measurement)


if __name__ == "__main__":
    main()
