# ppo_sparse.py
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 1) Воспроизводимость и девайс
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2) Среда со спарс-ревардом
# =========================
class DelayedBinaryEnv:
    """
    Игровая среда со следующими свойствами:
    - Наблюдение: x_t ∈ R^d
    - Действия: Discrete(D)
    - Длина эпизода: T (фиксированная)
    - Награда: в конце эпизода 1, если правильных действий >= K, иначе 0
    - "Правильность" задаётся скрытой матрицей W ∈ R^{d×D} и правилом argmax(W^T x_t)

    Это имитация "шахматоподобной" постановки без промежуточной обратной связи.
    """

    def __init__(self, d: int, D: int, T: int, K: int,
                 obs_scale: float = 1.0, w_scale: float = 1.0):
        assert D > d, "По условию задачи D должно быть > d"
        assert 1 <= K <= T
        self.d = d
        self.D = D
        self.T = T
        self.K = K
        self.obs_scale = obs_scale
        self.w_scale = w_scale

        self.t = 0
        self.W = None               # скрытая матрица W
        self.X_seq = None           # последовательность наблюдений
        self.correct_so_far = 0

    def reset(self) -> np.ndarray:
        # Случайная скрытая матрица W (меняется каждый эпизод — даёт разнообразие задач)
        self.W = np.random.randn(self.d, self.D) * self.w_scale

        # Предварительно сгенерим всю последовательность наблюдений на эпизод
        self.X_seq = np.random.randn(self.T, self.d) * self.obs_scale

        self.t = 0
        self.correct_so_far = 0

        return self.X_seq[self.t].astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.X_seq is not None, "Сначала вызовите reset()"
        x_t = self.X_seq[self.t]  # текущее наблюдение

        # Вычислим "правильное" действие, которое агенту не известно
        scores = x_t @ self.W  # shape: (D,)
        correct_action = int(np.argmax(scores))

        if action == correct_action:
            self.correct_so_far += 1

        self.t += 1
        done = (self.t >= self.T)

        if done:
            # Спарс-ревард: 1 если набрали хотя бы K "правильных" действий
            reward = 1.0 if self.correct_so_far >= self.K else 0.0
            # Начнём новый эпизод при следующем reset(); сейчас отдадим пустой obs
            next_obs = np.zeros(self.d, dtype=np.float32)
        else:
            reward = 0.0
            next_obs = self.X_seq[self.t].astype(np.float32)

        info = {
            "t": self.t,
            "correct_so_far": self.correct_so_far,
        }
        return next_obs, reward, done, info


# =========================
# 3) Векторизация среды (для стабильности PPO)
# =========================
class VectorizedEnv:
    """
    Простейший векторизатор поверх DelayedBinaryEnv без внешних зависимостей.
    """

    def __init__(self, make_env_fn, num_envs: int):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self) -> np.ndarray:
        obs = [env.reset() for env in self.envs]
        return np.stack(obs, axis=0)  # (num_envs, d)

    def step(self, actions: np.ndarray):
        next_obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            ob, r, d, info = env.step(int(a))
            next_obs.append(ob)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
            if d:
                # Автоматически начало нового эпизода, чтобы поток не останавливался
                ob = env.reset()
                next_obs[-1] = ob
        return (np.stack(next_obs, axis=0),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool),
                infos)


# =========================
# 4) Политика + Критик (общая «спина»)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, d: int, D: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = d
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.policy_head = nn.Linear(in_dim, D)
        self.value_head = nn.Linear(in_dim, 1)

        self.apply(self._orthogonal_init)

    @staticmethod
    def _orthogonal_init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value

    def act(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value


# =========================
# 5) Буфер для rollout'ов
# =========================
@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


# =========================
# 6) GAE: расчет преимуществ и таргетов
# =========================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones — тензоры формы (T, N) (временная ось, параллельные энвы).
    Возвращаем:
      advantages (T, N)
      returns (T, N) = advantages + values
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    next_value = torch.zeros(N, device=rewards.device)  # V(s_{T}) = 0 (эпизоды завершаются)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()  # если done=True, то обнуляем бустрапинг
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


# =========================
# 7) PPO-оптимизация
# =========================
@dataclass
class PPOConfig:
    d: int = 16
    D: int = 32
    T: int = 16          # длина эпизода
    K: int = 10          # порог "успеха"
    num_envs: int = 32   # параллельные среды
    total_updates: int = 1500
    steps_per_update: int = 16  # мы равняем на T
    minibatch_size: int = 1024
    ppo_epochs: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coeff: float = 0.5
    ent_coeff: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 42


def ppo_train(cfg: PPOConfig):
    set_seed(cfg.seed)

    # === Среда ===
    make_env_fn = lambda: DelayedBinaryEnv(d=cfg.d, D=cfg.D, T=cfg.T, K=cfg.K)
    venv = VectorizedEnv(make_env_fn, cfg.num_envs)
    obs = venv.reset()  # (N, d)

    # === Модель и оптимизатор ===
    model = ActorCritic(cfg.d, cfg.D).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Для отслеживания прогресса
    ema_success = 0.0
    ema_alpha = 0.02

    for update in range(1, cfg.total_updates + 1):
        # 1) Сбор траекторий длиной steps_per_update (равно T)
        mb_obs, mb_actions, mb_logprobs, mb_rewards, mb_dones, mb_values = [], [], [], [], [], []

        for step in range(cfg.steps_per_update):
            x = torch.tensor(obs, dtype=torch.float32, device=device)  # (N, d)
            with torch.no_grad():
                action, logprob, entropy, value = model.act(x)

            next_obs, rewards, dones, infos = venv.step(action.cpu().numpy())
            mb_obs.append(x)
            mb_actions.append(action)
            mb_logprobs.append(logprob)
            mb_rewards.append(torch.tensor(rewards, device=device))
            mb_dones.append(torch.tensor(dones, device=device))
            mb_values.append(value)

            obs = next_obs

        # Стекируем в тензоры формы (T, N, ...)
        obs_b = torch.stack(mb_obs)                       # (T, N, d)
        actions_b = torch.stack(mb_actions)               # (T, N)
        logprobs_b = torch.stack(mb_logprobs)             # (T, N)
        rewards_b = torch.stack(mb_rewards)               # (T, N)
        dones_b = torch.stack(mb_dones)                   # (T, N)
        values_b = torch.stack(mb_values)                 # (T, N)

        # 2) GAE
        advantages_b, returns_b = compute_gae(
            rewards_b, values_b, dones_b, gamma=cfg.gamma, lam=cfg.lam
        )
        # Нормализация преимуществ — стандартный приём
        adv_mean, adv_std = advantages_b.mean(), advantages_b.std(unbiased=False) + 1e-8
        advantages_b = (advantages_b - adv_mean) / adv_std

        # 3) Подготовка батчей для PPO (перемешаем и разобьём)
        T, N = cfg.steps_per_update, cfg.num_envs
        batch_size = T * N
        flat = lambda x: x.reshape(batch_size, *x.shape[2:])  # склеиваем T и N
        b_obs = flat(obs_b)
        b_actions = flat(actions_b)
        b_logprobs = flat(logprobs_b)
        b_advantages = flat(advantages_b)
        b_returns = flat(returns_b)
        b_values = flat(values_b)

        idxs = np.arange(batch_size)
        mb_size = cfg.minibatch_size
        if mb_size > batch_size:
            mb_size = batch_size

        # 4) PPO-оптимизация
        for epoch in range(cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                logits, values = model(b_obs[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()

                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_logprobs - b_logprobs[mb_idx])

                # Клипнутый и неклипнутый surrogate-объектив
                unclipped = ratio * b_advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_advantages[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss (можно также клиповать, но базовый вариант уже хорошо работает)
                value_pred = values.squeeze(-1)
                value_loss = 0.5 * (b_returns[mb_idx] - value_pred).pow(2).mean()

                # Общий лосс
                loss = policy_loss + cfg.vf_coeff * value_loss - cfg.ent_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # === Простейшая метрика "успешности": как часто получаем терминальный reward=1
        # За последний rollout rewards_b содержит 0 на всех шагах, кроме завершившихся эпизодов.
        # Посчитаем долю "успешных" эпизодов среди параллельных сред.
        # У нас эпизодная длина=T, значит в конце последнего шага есть конечные награды (0/1) в тех энвах, где done=True.
        terminal_rewards = rewards_b[-1].detach().cpu().numpy()  # (N,)
        success_rate = float((terminal_rewards > 0.5).mean())
        ema_success = (1 - ema_alpha) * ema_success + ema_alpha * success_rate

        if update % 10 == 0 or update == 1:
            print(f"[Update {update:4d}] success_rate={success_rate:.3f} (EMA={ema_success:.3f})  "
                  f"adv_std={adv_std.item():.3f}")

    return model


if __name__ == "__main__":
    cfg = PPOConfig(
        d=16,    # размер входного вектора наблюдения
        D=32,    # число дискретных действий (выходных логитов)
        T=16,    # длина эпизода
        K=10,    # порог для успеха (сколько шагов нужно угадать)
        num_envs=32,             # параллельных окружений (для стабильности)
        total_updates=600,       # можно увеличить для лучшей сходимости
        steps_per_update=16,     # = T, удобно собирать ровно эпизод
        minibatch_size=1024,     # размер минибатча для PPO
        ppo_epochs=4,            # проходов по данным на апдейт
        learning_rate=3e-4,
        clip_eps=0.2,
        vf_coeff=0.5,
        ent_coeff=0.01,
        seed=42,
    )
    trained_model = ppo_train(cfg)
