"""Побудова графіків для звіту за логами тренування та результатами тестів.

Вхід:
  results/train_log.csv          -- рядок на кожен тренувальний епізод
                                    (кроки, нагорода, довжина, epsilon, ознака раннього обриву);
  results/eval_log.csv           -- періодичні жадібні оцінки (16 епізодів кожні 50k кадрів)
                                    разом із поточними lr, loss та середнім Q;
  results/baseline_test_50.txt   -- вивід test_car_racing.py для бейзлайну;
  results/improved_test_50.txt   -- вивід test_car_racing.py для покращеної моделі.

Вихід (results/figures/):
  training_curve.png    -- нагороди тренувальних епізодів + розклад epsilon;
  early_truncation.png  -- частка епізодів, обірваних обгорткою NoProgressTruncation;
  eval_curve.png        -- результати жадібних оцінок (середнє ± std, мінімум);
  loss_q_curve.png      -- Huber-втрата та середнє Q(s, a) протягом тренування;
  test_comparison.png   -- розподіл нагород бейзлайну та покращеної моделі на 50 епізодах.
"""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")  # рендер у файл, без вікна
import matplotlib.pyplot as plt
import numpy as np

RES = "results"
FIG = os.path.join(RES, "figures")
BASELINE_CLAIMED = 840.6  # заявлений результат бейзлайну з умови завдання
MA_WINDOW = 100           # вікно ковзного середнього для тренувальних кривих


def read_csv(path):
    """Прочитати CSV-журнал у список словників (ключі -- назви колонок)."""
    with open(path) as f:
        return list(csv.DictReader(f))


def read_test(path):
    """Витягти нагороди епізодів з текстового виводу test_car_racing.py
    (рядки виду "Episode 7<TAB>Reward 912.3")."""
    with open(path) as f:
        return np.array([float(m) for m in re.findall(r"Episode \d+\s+Reward (-?[\d.]+)", f.read())])


def moving_average(x, k=MA_WINDOW):
    """Ковзне середнє з вікном k (довжина результату len(x) - k + 1)."""
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_training():
    """Рисунки training_curve.png та early_truncation.png за train_log.csv."""
    rows = read_csv(os.path.join(RES, "train_log.csv"))
    steps = np.array([int(r["env_steps"]) for r in rows]) / 1e6   # вісь X -- млн кадрів
    score = np.array([float(r["score"]) for r in rows])
    eps = np.array([float(r["epsilon"]) for r in rows])
    early = np.array([int(r["truncated_early"]) for r in rows])

    k = MA_WINDOW
    ma = moving_average(score)

    # --- нагороди тренувальних епізодів та epsilon (друга вісь Y) ---
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
    # Об'єднана легенда для обох осей
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curve.png"), dpi=150)
    plt.close(fig)

    # --- частка епізодів, обірваних достроково (ковзне середнє, %) ---
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(steps[k - 1:], moving_average(early) * 100, color="tab:purple")
    ax.set_xlabel("Кроки середовища, млн")
    ax.set_ylabel("Обірвані достроково, %")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "early_truncation.png"), dpi=150)
    plt.close(fig)


def plot_eval():
    """Рисунки eval_curve.png та loss_q_curve.png за eval_log.csv."""
    rows = read_csv(os.path.join(RES, "eval_log.csv"))
    steps = np.array([int(r["env_steps"]) for r in rows]) / 1e6
    mean = np.array([float(r["mean"]) for r in rows])
    std = np.array([float(r["std"]) for r in rows])
    mn = np.array([float(r["min"]) for r in rows])

    # --- жадібні оцінки: середнє, смуга ± std, мінімум, лінія бейзлайну ---
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(steps, mean, "o-", color="tab:green", label="Середня нагорода (16 епізодів, ε = 0)")
    ax.fill_between(steps, mean - std, mean + std, color="tab:green", alpha=0.2, label="± std")
    ax.plot(steps, mn, "--", color="tab:gray", lw=1, label="Мінімум")
    ax.axhline(BASELINE_CLAIMED, color="tab:red", ls=":", label=f"Бейзлайн (заявлено) {BASELINE_CLAIMED}")
    best = int(np.argmax(mean))  # саме цю точку збережено як checkpoint_..._best.pth
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

    # --- Huber-втрата та середнє Q (значення записані в момент кожної оцінки) ---
    loss = np.array([float(r["loss"]) for r in rows])
    q = np.array([float(r["mean_q"]) for r in rows])
    lr = np.array([float(r["lr"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, loss, "o-", color="tab:orange")
    axes[0].set_title("Huber-втрата (мінібатч)")
    axes[0].set_xlabel("Кроки середовища, млн")
    axes[0].grid(alpha=0.3)
    ax_lr = axes[0].twinx()
    ax_lr.plot(steps, lr, color="tab:gray", ls="--")
    ax_lr.set_ylabel("learning rate", color="tab:gray")
    axes[1].plot(steps, q, "o-", color="tab:blue")
    axes[1].set_title("Середнє Q(s, a) на мінібатчі")
    axes[1].set_xlabel("Кроки середовища, млн")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "loss_q_curve.png"), dpi=150)
    plt.close(fig)


def plot_test_comparison():
    """Рисунок test_comparison.png: box-plot та гістограма нагород на 50 тестових епізодах."""
    base = read_test(os.path.join(RES, "baseline_test_50.txt"))
    impr = read_test(os.path.join(RES, "improved_test_50.txt"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    # Ліворуч: "ящик з вусами" + окремі епізоди (з невеликим горизонтальним шумом)
    ax = axes[0]
    ax.boxplot([base, impr], tick_labels=["Бейзлайн", "Покращена"], widths=0.5, showmeans=True)
    rng = np.random.default_rng(0)  # фіксований seed -- однаковий рисунок при перезапуску
    for i, d in enumerate([base, impr], start=1):
        ax.plot(rng.normal(i, 0.05, len(d)), d, ".", alpha=0.5)
    ax.set_ylabel("Нагорода за епізод")
    ax.set_title("Розподіл нагород, 50 епізодів")
    ax.grid(alpha=0.3, axis="y")

    # Праворуч: накладені гістограми з однаковими інтервалами
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
