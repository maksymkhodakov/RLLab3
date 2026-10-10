"""Візуалізація спостереження CarRacing-v3 для звіту.

Будує рисунок: повний RGB-кадр середовища (96x96x3) та чотири кадри
після передобробки (grayscale + resize 84x84 + frame stack), які
отримує на вхід Q-мережа.
"""

import os

import matplotlib
matplotlib.use("Agg")  # рендер без дисплея
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from dqn_car_racing import make_env

OUT = os.path.join("results", "figures", "observation.png")


def main():
    # Сире середовище -- щоб показати оригінальний RGB-кадр
    raw = gym.make("CarRacing-v3", continuous=False)
    env = make_env()
    raw.reset(seed=7)
    env.reset(seed=7)
    # Проїжджаємо трохи вперед (газ), щоб пройти анімацію наближення камери
    # і щоб у стеку кадрів був помітний рух
    for t in range(70):
        a = 3 if t % 6 else 0
        rgb, *_ = raw.step(a)
        stack, *_ = env.step(a)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB 96×96×3 (оригінал)")
    for k in range(4):
        axes[k + 1].imshow(np.asarray(stack)[k], cmap="gray", vmin=0, vmax=255)
        axes[k + 1].set_title(f"Кадр t−{3 - k}: 84×84, сірий")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
