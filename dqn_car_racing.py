"""Покращений Deep Q-Learning для CarRacing-v3 (Box2D).

Середовище: https://gymnasium.farama.org/environments/box2d/car_racing/

Спостереження -- RGB-кадри 96x96x3. Функція ``make_env`` (НЕ ЗМІНЕНА відносно
бейзлайну) переводить їх у відтінки сірого, зменшує до 84x84 та складає
4 послідовні кадри в один тензор (4, 84, 84), щоб мережа "бачила" рух.
Дискретний простір дій має 5 дій: нічого не робити, кермо вліво,
кермо вправо, газ, гальмо.

Що змінено відносно бейзлайну (dqn_car_racing.py з репозиторію курсу):

1. Архітектура -- Dueling DQN: після згорткової частини мережа розділяється
   на дві "голови": V(s) (цінність стану) та A(s, a) (перевага дії).
   Q(s, a) = V(s) + A(s, a) - mean_a A(s, a).
2. Ціль навчання -- Double DQN: дію для наступного стану обирає онлайн-мережа,
   а оцінює її цільова (target) мережа. Це прибирає систематичну
   переоцінку Q-значень, властиву класичному max-оператору.
3. n-крокові повернення (n = 3): нагорода "протікає" до попередніх станів
   швидше, ніж при однокроковому TD-оновленні.
4. Коректна обробка обрізання епізоду за часом: якщо епізод закінчився через
   ліміт кроків (truncated), ми продовжуємо бутстрепінг з Q(s'), а не
   вважаємо стан термінальним (у бейзлайні done = terminated OR truncated).
5. Компактний replay-буфер: кожне спостереження зберігається лише один раз
   (наступний стан -- це просто наступна комірка того самого потоку
   середовища), тому при тій самій пам'яті буфер утричі більший (120k проти 50k).
6. Розклад epsilon за кроками (кусково-лінійний 1.0 -> 0.1 -> 0.01) замість
   мультиплікативного згасання за епізодами; косинусне зменшення learning rate;
   м'яке (Polyak) оновлення цільової мережі.
7. Лише для ТРЕНУВАЛЬНИХ середовищ -- ранній обрив епізоду, якщо машина
   довго не отримує позитивної нагороди (з'їхала з траси / стоїть). Це
   економить час симуляції і не змінює параметрів самого середовища:
   ``make_env`` і тестове середовище залишаються ідентичними бейзлайну.
8. Вибір найкращого чекпоінта за результатом жадібної (epsilon = 0) оцінки
   на 16 нових трасах, а не за ковзним середнім тренувальних епізодів з
   увімкненим дослідженням.
9. Крок векторного середовища виконується асинхронно паралельно з
   оновленнями мережі на GPU (MPS/CUDA), що пришвидшує навчання.

Тестовий скрипт test_car_racing.py імпортує звідси ``make_env``, ``QNetwork``
та ``device`` -- їхній інтерфейс збережено.
"""

import argparse
import csv
import math
import os
import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    RecordVideo,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Пристрої обчислень
# ---------------------------------------------------------------------------
# ``device`` імпортує тестовий скрипт, тому лишаємо ту саму логіку, що й у
# бейзлайні (CUDA, якщо є, інакше CPU). Для тестування (батч з одного кадру)
# CPU цілком достатньо.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Для тренування додатково використовуємо Apple MPS (GPU на Mac M1/M2),
# якщо CUDA недоступна: оновлення мережі на ньому ~9 разів швидші, ніж на CPU.
if torch.cuda.is_available():
    train_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    train_device = torch.device("mps")
else:
    train_device = torch.device("cpu")

torch.backends.cudnn.benchmark = True  # автопідбір згорток для фіксованого розміру входу

FRAME_STACK = 4   # скільки останніх кадрів подаємо мережі
FRAME_SIZE = 84   # розмір кадру після зменшення

RESULTS_DIR = "results"
CHECKPOINT_BEST = "checkpoint_CarRacing-v3_best.pth"
CHECKPOINT_LAST = "checkpoint_CarRacing-v3_last.pth"


# ---------------------------------------------------------------------------
# Середовище
# ---------------------------------------------------------------------------
def make_env(render_mode=None):
    """CarRacing з обгортками grayscale, resize та frame-stacking.

    УВАГА: функція залишена без змін відносно бейзлайну -- за умовою
    лабораторної параметри середовища змінювати не можна.
    """
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = FrameStackObservation(env, FRAME_STACK)
    return env


class NoProgressTruncation(gym.Wrapper):
    """Обгортка ЛИШЕ для тренування: обриває епізод, якщо агент ``patience``
    кроків поспіль не отримав жодної позитивної нагороди.

    У CarRacing позитивна нагорода надходить тільки за нові відвідані
    плитки траси. Якщо її довго немає -- машина з'їхала на траву, крутиться
    на місці або стоїть, і решта епізоду (до 1000 кадрів) дає лише
    штраф -0.1 за кадр та майже нічого не навчає. Обрив позначаємо як
    ``truncated`` (а не ``terminated``), тож у цілі навчання ми все одно
    бутстрепимося з Q(s') -- тобто це не змінює задачу, а лише економить
    симуляцію. Перші ~50 кадрів епізоду (анімація наближення камери)
    також не мають позитивних нагород, тому ``patience`` має бути більшим.
    """

    def __init__(self, env, patience=100):
        super().__init__(env)
        self.patience = patience
        self.no_progress = 0  # лічильник кроків без позитивної нагороди

    def reset(self, **kwargs):
        self.no_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0:
            self.no_progress = 0
        else:
            self.no_progress += 1
        if self.no_progress >= self.patience and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


def make_train_env():
    """Тренувальне середовище = незмінне ``make_env()`` + ранній обрив."""
    return NoProgressTruncation(make_env(), patience=100)


def make_vec_env(num_envs, train=True):
    """Кілька екземплярів CarRacing, що крокують паралельно в підпроцесах.

    Режим автоскидання NEXT_STEP (типовий у gymnasium 1.x): після кроку, на
    якому під-середовище завершилось, НАСТУПНИЙ виклик step() повертає
    початкове спостереження нового епізоду (дія на цьому кроці ігнорується).
    """
    fn = make_train_env if train else make_env
    return gym.vector.AsyncVectorEnv(
        [fn for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Dueling-CNN: стек кадрів -> Q-значення дій.

    Згорткова частина така сама, як у Nature DQN (Mnih et al., 2015):
    3 згорткові шари, вихід 64 x 7 x 7 = 3136 ознак. Далі -- дві окремі
    повнозв'язні гілки (Wang et al., 2016):
      * value-гілка  оцінює V(s) -- наскільки добрий сам стан;
      * advantage-гілка оцінює A(s, a) -- наскільки кожна дія краща за середню.
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a). Віднімання середнього робить
    розклад однозначним. Перевага: V(s) навчається на кожному переході
    незалежно від того, яку дію було обрано, що прискорює і стабілізує
    навчання, коли багато дій мають близьку цінність (як у CarRacing на
    прямих ділянках).
    """

    def __init__(self, in_channels=FRAME_STACK, a_size=5):
        super().__init__()
        self.a_size = a_size

        # Згорткова частина (ідентична бейзлайну)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)  # 84 -> 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)           # 20 -> 9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)           # 9 -> 7

        # Гілка цінності стану V(s)
        self.fc_value = nn.Linear(64 * 7 * 7, 512)
        self.value = nn.Linear(512, 1)
        # Гілка переваги дій A(s, a)
        self.fc_adv = nn.Linear(64 * 7 * 7, 512)
        self.advantage = nn.Linear(512, a_size)

    def forward(self, x):
        # Кадри зберігаються як uint8 [0, 255]; нормалізуємо до [0, 1] вже на пристрої
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        v = self.value(F.relu(self.fc_value(x)))        # (B, 1)
        a = self.advantage(F.relu(self.fc_adv(x)))      # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)      # (B, A)

    def _device(self):
        # Пристрій беремо з параметрів моделі, щоб act() працював і на CPU
        # (тестовий скрипт), і на MPS/CUDA (тренування).
        return next(self.parameters()).device

    def act(self, state, epsilon=0.0):
        """Epsilon-жадібна дія для одного стану (використовує тестовий скрипт)."""
        if np.random.rand() <= epsilon:
            return np.random.randint(self.a_size)
        state = torch.as_tensor(np.asarray(state), device=self._device()).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values, dim=1).item()

    def act_batch(self, states, epsilon=0.0):
        """Epsilon-жадібні дії для батчу станів (по одному на середовище)."""
        states = torch.as_tensor(np.asarray(states), device=self._device())
        with torch.no_grad():
            actions = self.forward(states).argmax(dim=1).cpu().numpy()
        explore = np.random.rand(len(actions)) < epsilon
        actions[explore] = np.random.randint(self.a_size, size=int(explore.sum()))
        return actions


# ---------------------------------------------------------------------------
# Буфер відтворення досвіду з n-кроковими поверненнями
# ---------------------------------------------------------------------------
class NStepReplayBuffer:
    """Кільцевий буфер, що зберігає ПОТІК спостережень кожного середовища.

    Бейзлайн зберігав для кожного переходу і state, і next_state, тобто
    кожен кадр лежав у пам'яті двічі. Тут для кожного з E середовищ є власна
    "стрічка" з C комірок; на кожному кроці векторного середовища в усі
    стрічки дописується рівно одне нове спостереження. Наступний стан
    переходу з комірки j -- це комірка j + 1 тієї ж стрічки, тож дублювання
    немає і в ту саму пам'ять уміщується вдвічі більше переходів.

    Для комірки j зберігаємо дію a_j, нагороду r_j, ознаки terminated_j та
    end_j (епізод закінчився після цього кроку: terminated або truncated), а
    також valid_j -- чи є перехід j -> j+1 справжнім кроком середовища.
    Невалідний перехід виникає на кроці автоскидання: спостереження j --
    останній кадр старого епізоду, j+1 -- перший кадр нового.

    Під час вибірки обчислюється n-крокове повернення
        R = r_j + g r_{j+1} + ... + g^{m-1} r_{j+m-1},
    де m <= n (менше, якщо епізод завершився раніше), а також коефіцієнт
    бутстрепу g^m (або 0, якщо епізод завершився термінально). Ціль:
        y = R + g^m * Q_target(s_{j+m}, argmax_a Q_online(s_{j+m}, a)).
    """

    def __init__(self, num_envs, capacity_per_env, obs_shape, n_step=3, gamma=0.99):
        self.E = num_envs
        self.C = capacity_per_env
        self.n = n_step
        self.gamma = gamma

        # uint8 -- щоб зменшити пам'ять і трафік CPU -> GPU у 4 рази
        self.obs = np.zeros((num_envs, capacity_per_env, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((num_envs, capacity_per_env), dtype=np.int64)
        self.rewards = np.zeros((num_envs, capacity_per_env), dtype=np.float32)
        self.terms = np.zeros((num_envs, capacity_per_env), dtype=bool)
        self.ends = np.zeros((num_envs, capacity_per_env), dtype=bool)
        self.valid = np.zeros((num_envs, capacity_per_env), dtype=bool)
        # t -- "абсолютний" індекс поточного (останнього записаного) спостереження;
        # фізична комірка = t % C
        self.t = 0

    def start(self, states):
        """Записати початкові спостереження після env.reset()."""
        self.t = 0
        self.obs[:, 0] = states

    def add(self, actions, rewards, terms, ends, valid, next_states):
        """Додати один крок векторного середовища (по переходу на кожне середовище)."""
        k = self.t % self.C
        self.actions[:, k] = actions
        self.rewards[:, k] = rewards
        self.terms[:, k] = terms
        self.ends[:, k] = ends
        self.valid[:, k] = valid

        k1 = (self.t + 1) % self.C
        self.obs[:, k1] = next_states
        # Метадані комірки k1 ще не записані (там могли лишитися старі дані)
        self.valid[:, k1] = False
        self.t += 1

    def _window(self):
        """Діапазон абсолютних індексів j, з яких можна почати n-кроковий перехід.

        Потрібні спостереження j..j+n та метадані j..j+n-1, які ще не
        перезаписані: j >= t - C + 1 та j + n <= t.
        """
        lo = max(0, self.t - self.C + 1)
        hi = self.t - self.n
        return lo, hi

    def __len__(self):
        lo, hi = self._window()
        return max(0, hi - lo + 1) * self.E

    def sample(self, batch_size, to_device):
        lo, hi = self._window()

        # Відбір випадкових (середовище, індекс) з відкиданням невалідних
        # переходів (кроків автоскидання). Їх лише ~0.1-1%, тому достатньо
        # одного-двох раундів.
        env_idx, j_idx = [], []
        need = batch_size
        while need > 0:
            e = np.random.randint(0, self.E, size=2 * need)
            j = np.random.randint(lo, hi + 1, size=2 * need)
            ok = self.valid[e, j % self.C]
            env_idx.append(e[ok][:need])
            j_idx.append(j[ok][:need])
            need -= len(env_idx[-1])
        e = np.concatenate(env_idx)
        j = np.concatenate(j_idx)

        # Векторизоване обчислення n-крокового повернення
        ret = np.zeros(batch_size, dtype=np.float32)
        alive = np.ones(batch_size, dtype=bool)        # епізод ще триває
        boot = np.ones(batch_size, dtype=np.float32)   # 0, якщо епізод завершився термінально
        disc_boot = np.ones(batch_size, dtype=np.float32)  # g^m для бутстрепу
        nxt = j.copy()                                 # абсолютний індекс s_{j+m}
        disc = 1.0
        for s in range(self.n):
            idx = (j + s) % self.C
            ret += np.where(alive, disc * self.rewards[e, idx], 0.0).astype(np.float32)
            nxt = np.where(alive, j + s + 1, nxt)
            disc_boot = np.where(alive, disc * self.gamma, disc_boot)
            boot = np.where(alive & self.terms[e, idx], 0.0, boot)
            alive = alive & ~self.ends[e, idx]
            disc *= self.gamma

        states = torch.from_numpy(self.obs[e, j % self.C]).to(to_device)
        next_states = torch.from_numpy(self.obs[e, nxt % self.C]).to(to_device)
        actions = torch.from_numpy(self.actions[e, j % self.C]).to(to_device).unsqueeze(1)
        returns = torch.from_numpy(ret).to(to_device).unsqueeze(1)
        bootstrap = torch.from_numpy((boot * disc_boot).astype(np.float32)).to(to_device).unsqueeze(1)
        return states, actions, returns, next_states, bootstrap


# ---------------------------------------------------------------------------
# Розклади гіперпараметрів
# ---------------------------------------------------------------------------
def piecewise_linear(progress, points):
    """Кусково-лінійна інтерполяція: points = [(x0, y0), (x1, y1), ...]."""
    if progress <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if progress <= x1:
            return y0 + (progress - x0) / (x1 - x0) * (y1 - y0)
    return points[-1][1]


# Epsilon: швидко 1.0 -> 0.1 за перші 20% кроків (активне дослідження),
# далі повільно 0.1 -> 0.01 до 60%, і 0.01 до кінця (майже жадібна політика).
EPSILON_SCHEDULE = [(0.0, 1.0), (0.2, 0.1), (0.6, 0.01)]


# ---------------------------------------------------------------------------
# Агент
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, action_dim, obs_shape, num_envs, lr=1e-4, lr_min=1e-5,
                 gamma=0.99, n_step=3, buffer_per_env=15000, batch_size=64,
                 tau=0.0025, total_updates=1):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.lr_min = lr_min
        self.total_updates = max(1, total_updates)
        self.epsilon = 1.0

        # batch_size = 64 і 2 оновлення на крок векторного середовища: та сама
        # кількість семплів, що й 1 x 128, але вдвічі більше градієнтних кроків
        # (на MPS батч 64 рахується ~1.6 раза швидше за 128).
        self.model = QNetwork(a_size=action_dim).to(train_device)
        self.target_model = QNetwork(a_size=action_dim).to(train_device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self.online_params = list(self.model.parameters())
        self.target_params = list(self.target_model.parameters())

        # eps=1.5e-4 -- стандартне значення Adam для DQN (Rainbow, Dopamine):
        # робить крок менш чутливим до дуже малих оцінок другого моменту
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1.5e-4)
        self.memory = NStepReplayBuffer(num_envs, buffer_per_env, obs_shape,
                                        n_step=n_step, gamma=gamma)
        self.learn_steps = 0
        self.last_loss = float("nan")
        self.last_q = float("nan")

    def _update_lr(self):
        """Косинусне згасання learning rate від lr до lr_min за все навчання.

        На початку великий крок швидко навчає, наприкінці малий крок
        "заспокоює" ваги, і політика менше коливається між чекпоінтами.
        """
        progress = min(1.0, self.learn_steps / self.total_updates)
        lr = self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def replay(self):
        """Один крок градієнтного спуску на мінібатчі з буфера."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, returns, next_states, bootstrap = self.memory.sample(
            self.batch_size, train_device)

        # Q(s, a) поточної мережі для дій, які реально виконувались
        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: argmax обирає ОНЛАЙН-мережа ...
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            # ... а значення цієї дії бере ЦІЛЬОВА мережа
            next_q = self.target_model(next_states).gather(1, next_actions)
            target = returns + bootstrap * next_q

        # Huber-втрата: квадратична для малих помилок, лінійна для великих --
        # стійка до рідкісних великих TD-помилок (наприклад, -100 за виїзд за поле)
        loss = F.smooth_l1_loss(q_values, target)

        self._update_lr()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1

        # М'яке оновлення цільової мережі: theta' <- tau*theta + (1-tau)*theta'.
        # Ціль змінюється плавно на кожному кроці, без "стрибків" раз на N кроків.
        # Одним "foreach"-викликом для всіх тензорів -- менше запусків ядер на GPU.
        # tau = 0.0025 -> стала часу ~400 оновлень (близько до жорсткого
        # оновлення раз на 1000 кроків у бейзлайні, але без стрибків).
        with torch.no_grad():
            torch._foreach_lerp_(self.target_params, self.online_params, self.tau)

        # Статистика для логів (рідко, бо .item() синхронізує GPU)
        if self.learn_steps % 500 == 0:
            self.last_loss = loss.item()
            self.last_q = q_values.mean().item()


# ---------------------------------------------------------------------------
# Оцінювання жадібної політики (для вибору чекпоінта)
# ---------------------------------------------------------------------------
def evaluate_policy_vec(model, eval_env, num_episodes=16):
    """Прогнати ``num_episodes`` повних епізодів жадібної політики (epsilon = 0)
    на НЕЗМІННОМУ середовищі ``make_env`` (без раннього обриву).

    Використовуємо векторне середовище: кожне з E під-середовищ проходить
    ceil(num_episodes / E) епізодів. Повертає масив нагород за епізоди.
    """
    num_envs = eval_env.num_envs
    per_env = math.ceil(num_episodes / num_envs)
    was_training = model.training
    model.eval()

    states, _ = eval_env.reset()
    ep_ret = np.zeros(num_envs)
    finished = np.zeros(num_envs, dtype=int)
    prev_done = np.zeros(num_envs, dtype=bool)
    results = []
    while (finished < per_env).any():
        actions = model.act_batch(states, epsilon=0.0)
        states, rewards, terms, truncs, _ = eval_env.step(actions)
        dones = np.logical_or(terms, truncs)
        for i in range(num_envs):
            if prev_done[i]:
                continue  # крок автоскидання -- не частина епізоду
            ep_ret[i] += rewards[i]
            if dones[i]:
                if finished[i] < per_env:
                    results.append(ep_ret[i])
                finished[i] += 1
                ep_ret[i] = 0.0
        prev_done = dones

    if was_training:
        model.train()
    return np.array(results[:num_episodes])


# ---------------------------------------------------------------------------
# Тренування
# ---------------------------------------------------------------------------
def train(num_envs=8, total_steps=1_500_000, updates_per_step=2, learning_starts=20_000,
          eval_every=50_000, eval_episodes=16, seed=42, results_dir=RESULTS_DIR):
    """Тренування з кількома паралельними середовищами CarRacing.

    Бюджет задається кількістю кроків середовища (кадрів), а не епізодів:
    через ранній обрив тренувальні епізоди мають різну довжину. Для
    порівняння: 1000 епізодів бейзлайну -- це ~1 000 000 кадрів.

    На кожній ітерації:
      1. агент обирає дії для всіх E середовищ (epsilon-жадібно);
      2. дії надсилаються в підпроцеси (step_async) і, ПОКИ середовища
         симулюють крок, виконується ``updates_per_step`` оновлень мережі на GPU;
      3. результати кроку (step_wait) записуються в буфер.
    """
    os.makedirs(results_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_vec_env(num_envs, train=True)
    eval_env = make_vec_env(num_envs, train=False)  # незмінне середовище для оцінки

    total_vec_steps = total_steps // num_envs
    total_updates = max(1, (total_vec_steps - learning_starts // num_envs) * updates_per_step)
    agent = DQNAgent(env.single_action_space.n, env.single_observation_space.shape,
                     num_envs=num_envs, total_updates=total_updates)
    print(f"train device: {train_device}, envs: {num_envs}, total steps: {total_steps}, "
          f"updates: {total_updates}")

    # Логи для графіків у звіті
    train_log = open(os.path.join(results_dir, "train_log.csv"), "w", newline="")
    train_writer = csv.writer(train_log)
    train_writer.writerow(["episode", "env_steps", "score", "length", "epsilon", "truncated_early",
                           "wall_time"])
    eval_log = open(os.path.join(results_dir, "eval_log.csv"), "w", newline="")
    eval_writer = csv.writer(eval_log)
    eval_writer.writerow(["env_steps", "episodes", "mean", "std", "min", "max", "lr", "loss",
                          "mean_q", "wall_time"])

    scores_deque = deque(maxlen=100)
    best_eval = -np.inf
    start_time = time.time()

    states, _ = env.reset(seed=seed)
    agent.memory.start(states)
    prev_done = np.zeros(num_envs, dtype=bool)
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=int)
    completed = 0
    env_steps = 0
    next_eval = eval_every

    for vec_step in range(total_vec_steps):
        agent.epsilon = piecewise_linear(env_steps / total_steps, EPSILON_SCHEDULE)
        actions = agent.model.act_batch(states, epsilon=agent.epsilon)

        # Середовища крокують у підпроцесах паралельно з навчанням мережі
        env.step_async(actions)
        if env_steps >= learning_starts:
            for _ in range(updates_per_step):
                agent.replay()
        next_states, rewards, terms, truncs, _ = env.step_wait()
        dones = np.logical_or(terms, truncs)

        # Перехід валідний, якщо попередній крок не завершив епізод
        # (інакше це крок автоскидання: next_states -- початок нового епізоду)
        agent.memory.add(actions, rewards, terms, dones, ~prev_done, next_states)

        for i in range(num_envs):
            if prev_done[i]:
                continue
            env_steps += 1
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            if dones[i]:
                score = episode_rewards[i]
                completed += 1
                scores_deque.append(score)
                early = bool(truncs[i] and episode_lengths[i] < 1000)
                train_writer.writerow([completed, env_steps, f"{score:.2f}", episode_lengths[i],
                                       f"{agent.epsilon:.4f}", int(early),
                                       f"{time.time() - start_time:.1f}"])
                if completed % 10 == 0:
                    train_log.flush()
                    print(f"Episode {completed}\tSteps {env_steps}\tScore {score:.1f}\t"
                          f"Avg100 {np.mean(scores_deque):.1f}\tEps {agent.epsilon:.3f}\t"
                          f"Loss {agent.last_loss:.3f}\tQ {agent.last_q:.1f}\t"
                          f"t {(time.time() - start_time) / 60:.1f} min")
                episode_rewards[i] = 0
                episode_lengths[i] = 0

        states = next_states
        prev_done = dones

        # Періодична жадібна оцінка на справжньому (тестовому) середовищі
        if env_steps >= next_eval or vec_step == total_vec_steps - 1:
            next_eval += eval_every
            res = evaluate_policy_vec(agent.model, eval_env, eval_episodes)
            lr = agent.optimizer.param_groups[0]["lr"]
            eval_writer.writerow([env_steps, len(res), f"{res.mean():.2f}", f"{res.std():.2f}",
                                  f"{res.min():.2f}", f"{res.max():.2f}", f"{lr:.2e}",
                                  f"{agent.last_loss:.4f}", f"{agent.last_q:.2f}",
                                  f"{time.time() - start_time:.1f}"])
            eval_log.flush()
            msg = (f"  [EVAL] steps {env_steps}: mean {res.mean():.1f} std {res.std():.1f} "
                   f"min {res.min():.1f} max {res.max():.1f}")
            if res.mean() > best_eval:
                best_eval = res.mean()
                torch.save({k: v.cpu() for k, v in agent.model.state_dict().items()},
                           CHECKPOINT_BEST)
                msg += f"  -> new best, saved {CHECKPOINT_BEST}"
            print(msg)

    torch.save({k: v.cpu() for k, v in agent.model.state_dict().items()}, CHECKPOINT_LAST)
    train_log.close()
    eval_log.close()
    env.close()
    eval_env.close()
    print(f"Done in {(time.time() - start_time) / 60:.1f} min. Best eval mean {best_eval:.1f}")


# ---------------------------------------------------------------------------
# Перегляд / запис відео (як у бейзлайні)
# ---------------------------------------------------------------------------
def test(env, policy, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    avg = total_reward / num_episodes
    print(f"Total Reward: {avg:.1f}")
    return avg


def render(checkpoint=CHECKPOINT_BEST, num_episodes=3, video_folder="videos"):
    """Записати відео гри натренованого агента у MP4 (працює без дисплея)."""
    env = make_env(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="car_racing",
        episode_trigger=lambda e: True,
    )
    policy = QNetwork(a_size=env.action_space.n).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()
    test(env, policy, num_episodes=num_episodes)
    env.close()
    print(f"Saved videos to {video_folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved DQN for CarRacing-v3")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--render", action="store_true", help="лише записати відео з чекпоінта")
    args = parser.parse_args()

    if args.render:
        render()
    else:
        train(num_envs=args.num_envs, total_steps=args.total_steps,
              updates_per_step=args.updates_per_step, eval_every=args.eval_every,
              eval_episodes=args.eval_episodes, seed=args.seed, results_dir=args.results_dir)
