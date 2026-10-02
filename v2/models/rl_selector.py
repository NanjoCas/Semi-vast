"""
RLSelector: PPO-based Reinforced Selector for pseudo-label quality selection.

Frames the pseudo-label filtering problem as a sequential decision process:
for each unlabeled sample in the pseudo-labeled pool the agent observes a
4-dimensional state vector and decides whether to discard (0) or keep (1) the
sample.  After a full episode the agent receives a reward that combines the
downstream F1 improvement and the mean logical consistency of selected samples.

State  : [confidence, entropy, |LogicScore|, diversity]   shape (4,)
Action : Discrete(2)  — 0 = discard, 1 = keep
Reward : R = alpha * delta_F1 + beta * mean(|LogicScore|_selected)

v2: see PPOSelector docstring for the changed training/selection procedure.
"""

import math
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Policy + Value network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Shared-trunk MLP with a policy head and a value head.

    Trunk  : Linear(4, 64) -> ReLU -> Linear(64, 32) -> ReLU
    Policy : Linear(32, 2)   — raw action logits
    Value  : Linear(32, 1)   — scalar state-value estimate
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, action_dim)
        self.value_head = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        # Smaller gain for output heads (common PPO practice)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: float tensor of shape (..., state_dim)

        Returns:
            action_logits : tensor of shape (..., action_dim)
            value         : tensor of shape (..., 1)
        """
        features = self.trunk(x)
        return self.policy_head(features), self.value_head(features)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-size ring buffer that accumulates one or more episodes of experience
    and computes GAE-lambda returns and advantages on demand.
    """

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        """Reset the buffer to empty."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        """
        Compute discounted returns and GAE advantages in-place.

        Args:
            gamma      : discount factor
            gae_lambda : GAE smoothing parameter
            last_value : bootstrap value for the state after the last stored
                         transition (0.0 if the episode ended naturally)
        """
        n = len(self.rewards)
        self.returns = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
            next_value = self.values[t]

        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    # ------------------------------------------------------------------
    # Tensor export
    # ------------------------------------------------------------------

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Return all stored data as a dict of float32 tensors on CPU.

        Requires that ``compute_returns_and_advantages`` has been called first.
        """
        return {
            "states":     torch.tensor(np.array(self.states),    dtype=torch.float32),
            "actions":    torch.tensor(np.array(self.actions),   dtype=torch.long),
            "log_probs":  torch.tensor(np.array(self.log_probs), dtype=torch.float32),
            "returns":    torch.tensor(self.returns,             dtype=torch.float32),
            "advantages": torch.tensor(self.advantages,          dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# PPO Selector
# ---------------------------------------------------------------------------

class PPOSelector:
    """
    PPO-based agent that selects high-quality pseudo-labeled samples from a
    noisy pool for semi-supervised fake-news detection.

    v2 training procedure (replaces v1's single pass over the pool):
      1. baseline_f1 = probe F1 with labeled data only; it stays fixed.
         (v1 replaced it with the previous episode's F1, so delta_F1 compared
         two unrelated random subsets.)
      2. For n_iterations: sample episodes_per_iter random episodes of
         episode_size samples, roll out the current policy, give each episode
         R = alpha * (F1(selected) - baseline_f1) + beta * mean|LogicScore|,
         then do one PPO update over all episodes. Several episodes per update
         give the advantage estimate a contrast between good and bad subsets.
      3. Final selection: one deterministic pass (keep iff P(keep|s) > 0.5) over
         the whole pool with the trained policy. v1 instead returned the
         stochastic picks accumulated during training, including the early
         near-uniform policy.
      4. Diversity uses the samples' real [CLS] embeddings when provided
         (v1 fell back to the 2-d [confidence, entropy] vector).

    Args:
        state_dim     : dimensionality of the state vector (default 4)
        action_dim    : number of discrete actions (default 2)
        lr            : Adam learning rate
        clip_epsilon  : PPO clipping range
        gamma         : discount factor (1.0 for terminal episode rewards)
        gae_lambda    : GAE lambda
        ppo_epochs    : number of gradient passes per PPO update
        batch_size    : mini-batch size for PPO update
        alpha         : weight for delta_F1 in the reward
        beta          : weight for mean |LogicScore| in the reward
        seed          : RNG seed for episode sampling and action sampling
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        ppo_epochs: int = 4,
        batch_size: int = 256,
        alpha: float = 0.7,
        beta: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        # A 4-input MLP is faster on CPU than paying a GPU round-trip per step.
        self.device = torch.device("cpu")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    @staticmethod
    def _static_features(pool: list) -> np.ndarray:
        """[confidence, entropy, |logic_score|] for every sample, shape (N, 3)."""
        return np.array(
            [
                [
                    float(s.get("confidence", 0.5)),
                    float(s.get("entropy", 0.0)),
                    abs(float(s.get("logic_score", 0.0))),
                ]
                for s in pool
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _unit_embeddings(pool: list, embeddings: Optional[np.ndarray]) -> np.ndarray:
        """Row-normalised embeddings used for diversity (falls back to [conf, entropy])."""
        if embeddings is None:
            embeddings = np.array(
                [[float(s.get("confidence", 0.5)), float(s.get("entropy", 0.0))] for s in pool],
                dtype=np.float32,
            )
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norms, 1e-8, None)

    @staticmethod
    def _diversity(unit_emb: np.ndarray, selected_sum: np.ndarray, n_selected: int) -> float:
        """Cosine distance between a sample and the centroid of selected samples."""
        if n_selected == 0:
            return 1.0
        norm = float(np.linalg.norm(selected_sum))
        if norm < 1e-8:
            return 1.0
        cos_sim = float(np.dot(unit_emb, selected_sum) / norm)
        return 1.0 - float(np.clip(cos_sim, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Action sampling (uses old_policy for data collection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_action(self, state: np.ndarray):
        state_t = torch.from_numpy(state).unsqueeze(0)
        logits, value = self.old_policy(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.squeeze(-1).item()),
        )

    @torch.no_grad()
    def keep_probability(self, state: np.ndarray) -> float:
        logits, _ = self.policy(torch.from_numpy(state).unsqueeze(0))
        return float(torch.softmax(logits, dim=-1)[0, 1].item())

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        selected: list,
        baseline_f1: float,
        val_f1_fn: Callable[[list], float],
    ) -> tuple:
        """R = alpha * (F1(selected) - baseline_f1) + beta * mean|LogicScore|_selected."""
        if not selected:
            return 0.0, baseline_f1, 0.0
        new_f1 = val_f1_fn(selected)
        mean_logic = float(np.mean([abs(float(s.get("logic_score", 0.0))) for s in selected]))
        reward = self.alpha * (new_f1 - baseline_f1) + self.beta * mean_logic
        return reward, new_f1, mean_logic

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, buffer: RolloutBuffer) -> None:
        """
        Perform ``ppo_epochs`` passes of PPO mini-batch updates.

        Loss = -L_CLIP + 0.5 * value_loss - 0.01 * entropy_bonus
        """
        data = buffer.get()

        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_lp = data["log_probs"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]

                logits, values = self.policy(states[idx])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(actions[idx])
                entropy = dist.entropy()

                ratio = torch.exp(new_lp - old_lp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns[idx])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    # Main selection loop
    # ------------------------------------------------------------------

    @staticmethod
    def _kept_record(sample: dict) -> dict:
        return {
            "id": sample.get("id"),
            "claim": sample.get("claim", ""),
            "evidence": sample.get("evidence", []),
            "pseudo_label": sample.get("pseudo_label"),
            "weight": float(sample.get("weight", sample.get("confidence", 1.0))),
            "logic_score": sample.get("logic_score", 0.0),
            "confidence": sample.get("confidence", 0.5),
            "source": sample.get("source", "unknown"),
        }

    def _rollout(
        self,
        pool: list,
        indices: np.ndarray,
        static: np.ndarray,
        unit_emb: np.ndarray,
    ) -> tuple:
        """Roll out the old policy over one episode; returns (kept samples, transitions)."""
        selected_sum = np.zeros(unit_emb.shape[1], dtype=np.float32)
        n_selected = 0
        kept: list = []
        transitions: list = []
        for i in indices:
            diversity = self._diversity(unit_emb[i], selected_sum, n_selected)
            state = np.append(static[i], np.float32(diversity)).astype(np.float32)
            action, log_prob, value = self._sample_action(state)
            if action == 1:
                kept.append(pool[i])
                selected_sum += unit_emb[i]
                n_selected += 1
            transitions.append((state, action, log_prob, value))
        return kept, transitions

    def select(
        self,
        pseudo_labeled_pool: list,
        val_f1_fn: Callable[[list], float],
        embeddings: Optional[np.ndarray] = None,
        n_iterations: int = 30,
        episodes_per_iter: int = 4,
        episode_size: int = 512,
        min_keep_ratio: float = 0.05,
        show_progress: bool = True,
    ) -> tuple:
        """
        Train the policy on random episodes, then select from the full pool.

        Args:
            pseudo_labeled_pool : list of dicts with ``confidence``, ``entropy``,
                                  ``logic_score``, ``pseudo_label``, ``weight``
            val_f1_fn           : callable(selected_list) -> probe macro-F1
            embeddings          : optional (N, D) array aligned with the pool,
                                  used for the diversity feature
            n_iterations        : number of PPO updates
            episodes_per_iter   : episodes collected per update
            episode_size        : samples per episode
            min_keep_ratio      : if the final deterministic pass keeps fewer
                                  than this fraction, top-up by keep probability

        Returns:
            (kept_records, info) where info holds the training history and
            selection statistics.
        """
        pool = list(pseudo_labeled_pool)
        if not pool:
            return [], {"history": [], "warning": "empty pool"}

        n = len(pool)
        static = self._static_features(pool)
        unit_emb = self._unit_embeddings(pool, embeddings)
        ep_size = min(int(episode_size), n)

        baseline_f1 = float(val_f1_fn([]))
        history: list = []

        it_iter = range(1, n_iterations + 1)
        if show_progress:
            it_iter = tqdm(it_iter, desc="PPO iterations", unit="it", dynamic_ncols=True)

        for it in it_iter:
            self.buffer.clear()
            ep_stats = []
            for _ in range(episodes_per_iter):
                indices = self.rng.choice(n, size=ep_size, replace=False)
                kept, transitions = self._rollout(pool, indices, static, unit_emb)
                reward, new_f1, mean_logic = self._compute_reward(kept, baseline_f1, val_f1_fn)
                last = len(transitions) - 1
                for t, (state, action, log_prob, value) in enumerate(transitions):
                    done = t == last
                    self.buffer.add(state, action, reward if done else 0.0, log_prob, value, done)
                ep_stats.append({
                    "reward": reward,
                    "f1": new_f1,
                    "delta_f1": new_f1 - baseline_f1,
                    "mean_abs_logic": mean_logic,
                    "keep_ratio": len(kept) / ep_size,
                })

            self.buffer.compute_returns_and_advantages(
                gamma=self.gamma, gae_lambda=self.gae_lambda, last_value=0.0
            )
            self._ppo_update(self.buffer)

            record = {
                "iteration": it,
                **{k: float(np.mean([e[k] for e in ep_stats])) for k in ep_stats[0]},
                "reward_std": float(np.std([e["reward"] for e in ep_stats])),
            }
            history.append(record)
            if show_progress and hasattr(it_iter, "set_postfix"):
                it_iter.set_postfix(
                    reward=f"{record['reward']:.4f}",
                    dF1=f"{record['delta_f1']:+.4f}",
                    keep=f"{record['keep_ratio']:.2f}",
                )

        # ---- Final deterministic pass over the full pool -----------------
        order = self.rng.permutation(n)
        selected_sum = np.zeros(unit_emb.shape[1], dtype=np.float32)
        n_selected = 0
        keep_probs = np.zeros(n, dtype=np.float32)
        kept_mask = np.zeros(n, dtype=bool)
        for i in order:
            diversity = self._diversity(unit_emb[i], selected_sum, n_selected)
            state = np.append(static[i], np.float32(diversity)).astype(np.float32)
            p_keep = self.keep_probability(state)
            keep_probs[i] = p_keep
            if p_keep > 0.5:
                kept_mask[i] = True
                selected_sum += unit_emb[i]
                n_selected += 1

        warning = None
        min_keep = int(math.ceil(min_keep_ratio * n))
        if kept_mask.sum() < min_keep:
            warning = (
                f"policy kept {int(kept_mask.sum())}/{n} samples (< min_keep_ratio={min_keep_ratio}); "
                f"topped up to {min_keep} by keep probability"
            )
            kept_mask[np.argsort(-keep_probs, kind="stable")[:min_keep]] = True

        kept_records = [self._kept_record(pool[i]) for i in range(n) if kept_mask[i]]
        final_f1 = float(val_f1_fn(kept_records)) if kept_records else baseline_f1
        info = {
            "baseline_f1": baseline_f1,
            "final_selection_f1": final_f1,
            "final_delta_f1": final_f1 - baseline_f1,
            "pool_size": n,
            "selected": len(kept_records),
            "selection_ratio": len(kept_records) / n,
            "keep_prob_mean": float(keep_probs.mean()),
            "keep_prob_std": float(keep_probs.std()),
            "warning": warning,
            "history": history,
        }
        return kept_records, info

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the current policy weights and optimiser state.

        Args:
            path : file path ending in ``.pt`` or ``.pth``
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "policy_state_dict":    self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Restore policy weights and optimiser state from disk.

        Args:
            path : file path previously written by :meth:`save`
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.old_policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.old_policy.eval()
