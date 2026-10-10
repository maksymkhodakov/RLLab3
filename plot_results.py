"""Побудова графіків для звіту за логами тренування та результатами тестів.

Вхід:
  results/train_log.csv          -- кожен тренувальний епізод
  results/eval_log.csv           -- періодичні жадібні оцінки (16 епізодів)
  results/baseline_test_50.txt   -- вивід test_car_racing.py для бейзлайну
  results/improved_test_50.txt   -- вивід test_car_racing.py для покращеної моделі
Вихід: PNG-файли в results/figures/
"""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "results"
FIG = os.path.join(RES, "figures")
BASELINE_CLAIMED = 840.6  # заявлений результат бейзлайну з умови завдання


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_test(path):
    """Витягти нагороди епізодів з виводу test_car_racing.py."""
    with open(path) as f:
        return np.array([float(m) for m in re.findall(r"Episode \d+\s+Reward (-?[\d.]+)", f.read())])


def plot_training():
    rows = read_csv(os.path.join(RES, "train_log.csv"))
    steps = np.array([int(r["env_steps"]) for r in rows]) / 1e6
    score = np.array([float(r["score"]) for r in rows])
    eps = np.array([float(r["epsilon"]) for r in rows])
    early = np.array([int(r["truncated_early"]) for r in rows])

    # Ковзне середнє за 100 епізодів
    k = 100
    ma = np.convolve(score, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(steps, score, ".", ms=2, alpha=0.25, color="tab:blue", label="Нагорода епізоду")
    ax.plot(steps[k - 1:], ma, color="tab:blue", lw=2, label="Ковзне середнє (100 епізодів)")
    ax.set_xlabel("Кроки середовища, млн")
    ax.set_ylabel("Нагорода (тренувальний епізод)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, eps, color="tab:red", lw=1.5, label="epsilon")
    ax2.set_ylabel("epsilon", color="tab:red")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curve.png"), dpi=150)
    plt.close(fig)

    # Частка епізодів, обірваних достроково (обгортка NoProgressTruncation)
    ma_early = np.convolve(early, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(steps[k - 1:], ma_early * 100, color="tab:purple")
    ax.set_xlabel("Кроки середовища, млн")
    ax.set_ylabel("Обірвані достроково, %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "early_truncation.png"), dpi=150)
    plt.close(fig)


def plot_eval():
    rows = read_csv(os.path.join(RES, "eval_log.csv"))
    steps = np.array([int(r["env_steps"]) for r in rows]) / 1e6
    mean = np.array([float(r["mean"]) for r in rows])
    std = np.array([float(r["std"]) for r in rows])
    mn = np.array([float(r["min"]) for r in rows])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(steps, mean, "o-", color="tab:green", label="Середня нагорода (16 епізодів, ε = 0)")
    ax.fill_between(steps, mean - std, mean + std, color="tab:green", alpha=0.2, label="± std")
    ax.plot(steps, mn, "--", color="tab:gray", lw=1, label="Мінімум")
    ax.axhline(BASELINE_CLAIMED, color="tab:red", ls=":", label=f"Бейзлайн (заявлено) {BASELINE_CLAIMED}")
    best = int(np.argmax(mean))
    ax.plot(steps[best], mean[best], "*", ms=16, color="gold", mec="k",
            label=f"Найкращий чекпоінт: {mean[best]:.1f}")
    ax.set_xlabel("Кроки середовища, млн")
    ax.set_ylabel("Нагорода")
    ax.set_ylim(-150, 1000)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eval_curve.png"), dpi=150)
    plt.close(fig)


def plot_test_comparison():
    base = read_test(os.path.join(RES, "baseline_test_50.txt"))
    impr = read_test(os.path.join(RES, "improved_test_50.txt"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    ax.boxplot([base, impr], tick_labels=["Бейзлайн", "Покращена"], widths=0.5, showmeans=True)
    for i, d in enumerate([base, impr], start=1):
        ax.plot(np.random.normal(i, 0.05, len(d)), d, ".", alpha=0.5)
    ax.set_ylabel("Нагорода за епізод")
    ax.set_title("Розподіл нагород, 50 епізодів")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    bins = np.linspace(min(base.min(), impr.min()) - 10, 950, 30)
    ax.hist(base, bins=bins, alpha=0.6, label=f"Бейзлайн: {base.mean():.1f} ± {base.std():.1f}")
    ax.hist(impr, bins=bins, alpha=0.6, label=f"Покращена: {impr.mean():.1f} ± {impr.std():.1f}")
    ax.set_xlabel("Нагорода за епізод")
    ax.set_ylabel("Кількість епізодів")
    ax.set_title("Гістограма нагород")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "test_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    plot_training()
    plot_eval()
    if os.path.exists(os.path.join(RES, "improved_test_50.txt")):
        plot_test_comparison()
    print("figures saved to", FIG)
