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

    Args:
        state_dim     : dimensionality of the state vector (default 4)
        action_dim    : number of discrete actions (default 2)
        lr            : Adam learning rate
        clip_epsilon  : PPO clipping range
        gamma         : discount factor
        gae_lambda    : GAE lambda
        ppo_epochs    : number of gradient steps per PPO update
        batch_size    : mini-batch size for PPO update
        alpha         : weight for delta_F1 in the reward
        beta          : weight for mean |LogicScore| in the reward
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> None:
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Current policy (updated) and old policy (fixed during update step)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def compute_state(
        self,
        sample: dict,
        selected_embeddings: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Build the 4-dimensional state vector for a single sample.

        Components
        ----------
        confidence  : float in [0, 1] — model's max pseudo-label probability
        entropy     : float >= 0      — prediction entropy (nats)
        |logic_score| : float in [0, 1] — absolute logical consistency score
        diversity   : float in [0, 1] — cosine distance from selected centroid
                      (1.0 when no samples have been selected yet)

        Args:
            sample             : dict with keys ``confidence``, ``entropy``,
                                 ``logic_score``, and optionally ``embedding``
            selected_embeddings: list of embedding arrays for already-selected
                                 samples (may be None or empty)

        Returns:
            np.ndarray of shape (4,) and dtype float32
        """
        confidence  = float(sample.get("confidence", 0.5))
        entropy     = float(sample.get("entropy", 0.0))
        logic_score = abs(float(sample.get("logic_score", 0.0)))

        # ------ diversity ------------------------------------------------
        if not selected_embeddings:
            diversity = 1.0
        else:
            # Resolve sample embedding: use provided vector or proxy
            if "embedding" in sample and sample["embedding"] is not None:
                emb = np.asarray(sample["embedding"], dtype=np.float32)
            else:
                # Fallback proxy: two-dimensional vector [confidence, entropy]
                emb = np.array([confidence, entropy], dtype=np.float32)

            # Centroid of already-selected embeddings (in same space)
            centroid = np.mean(
                [np.asarray(e, dtype=np.float32) for e in selected_embeddings],
                axis=0,
            )

            # Cosine similarity -> distance
            norm_emb     = np.linalg.norm(emb)
            norm_centroid = np.linalg.norm(centroid)

            if norm_emb < 1e-8 or norm_centroid < 1e-8:
                diversity = 1.0
            else:
                cos_sim   = np.dot(emb, centroid) / (norm_emb * norm_centroid)
                cos_sim   = float(np.clip(cos_sim, -1.0, 1.0))
                diversity = 1.0 - cos_sim

        return np.array([confidence, entropy, logic_score, diversity], dtype=np.float32)

    # ------------------------------------------------------------------
    # Action sampling (uses old_policy for data collection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_action(self, state: np.ndarray):
        """
        Sample an action from the old policy.

        Returns:
            action   (int)
            log_prob (float)
            value    (float)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, value = self.old_policy(state_t)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.squeeze(-1).item()),
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        selected: list,
        baseline_f1: float,
        val_f1_fn: Callable[[list], float],
    ) -> tuple[float, float]:
        """
        Compute episode reward and return the new F1 used in the reward.

        R = alpha * delta_F1 + beta * mean(|LogicScore|_selected)

        Returns:
            (reward, new_f1)
        """
        if not selected:
            # No samples kept: keep baseline unchanged.
            return 0.0, baseline_f1

        new_f1 = val_f1_fn(selected)
        delta_f1 = new_f1 - baseline_f1

        logic_scores = [abs(float(s.get("logic_score", 0.0))) for s in selected]
        mean_logic = float(np.mean(logic_scores))

        reward = self.alpha * delta_f1 + self.beta * mean_logic
        return reward, new_f1

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, buffer: RolloutBuffer) -> None:
        """
        Perform ``ppo_epochs`` passes of PPO mini-batch updates.

        Loss = -L_CLIP + 0.5 * value_loss - 0.01 * entropy_bonus

        The old policy is synchronised with the current policy after all
        gradient steps are complete.
        """
        data = buffer.get()

        states     = data["states"].to(self.device)
        actions    = data["actions"].to(self.device)
        old_lp     = data["log_probs"].to(self.device)
        returns    = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]

        for _ in range(self.ppo_epochs):
            # Shuffle indices each epoch
            indices = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]

                b_states     = states[idx]
                b_actions    = actions[idx]
                b_old_lp     = old_lp[idx]
                b_returns    = returns[idx]
                b_advantages = advantages[idx]

                logits, values = self.policy(b_states)
                dist      = Categorical(logits=logits)
                new_lp    = dist.log_prob(b_actions)
                entropy   = dist.entropy()

                # Importance-sampling ratio
                ratio = torch.exp(new_lp - b_old_lp)

                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values.squeeze(-1), b_returns)

                # Entropy bonus (encourage exploration)
                entropy_bonus = entropy.mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        # Sync old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    # Main selection loop
    # ------------------------------------------------------------------

    def select(
        self,
        pseudo_labeled_pool: list,
        val_f1_fn: Callable[[list], float],
        n_steps: int = 512,
        show_progress: bool = True,
    ) -> list:
        """
        Run the PPO selection process over the full pseudo-labeled pool.

        The pool is processed in episodes of up to ``n_steps`` samples each.
        After each episode the PPO objective is optimised using the collected
        rollout.  The method returns all samples for which the agent chose
        action 1 (keep) in the *last* pass through the pool (the most refined
        policy).

        Args:
            pseudo_labeled_pool : list of dicts, each containing at minimum
                                  ``id``, ``claim``, ``pseudo_label``,
                                  ``confidence``, ``entropy``, ``logic_score``,
                                  and optionally ``embedding`` and ``weight``
            val_f1_fn           : callable(selected_list) -> float F1 score
            n_steps             : maximum number of samples per episode

        Returns:
            List of kept sample dicts, each guaranteed to contain the keys
            ``id``, ``claim``, ``pseudo_label``, and ``weight``.
        """
        if not pseudo_labeled_pool:
            return []

        pool_size = len(pseudo_labeled_pool)

        # Baseline F1 before any selection (empty selection → caller-defined)
        baseline_f1 = val_f1_fn([])

        # Split pool into episodes
        episode_boundaries = list(range(0, pool_size, n_steps)) + [pool_size]
        n_episodes = len(episode_boundaries) - 1

        # We store the keep-decisions from the last pass for the final return
        final_kept: list = []

        ep_iter = range(n_episodes)
        if show_progress:
            ep_iter = tqdm(ep_iter, desc="PPO episodes", unit="ep", dynamic_ncols=True)

        for ep_idx in ep_iter:
            ep_start = episode_boundaries[ep_idx]
            ep_end   = episode_boundaries[ep_idx + 1]
            episode_pool = pseudo_labeled_pool[ep_start:ep_end]

            self.buffer.clear()

            selected_this_episode: list = []
            selected_embeddings:   list = []
            episode_new_f1 = baseline_f1
            episode_reward = 0.0

            # ---- rollout ------------------------------------------------
            for step_idx, sample in enumerate(episode_pool):
                state = self.compute_state(sample, selected_embeddings)

                action, log_prob, value = self._sample_action(state)

                is_last_step = (step_idx == len(episode_pool) - 1)

                if action == 1:
                    kept_sample = {
                        "id":           sample.get("id", step_idx + ep_start),
                        "claim":        sample.get("claim", ""),
                        "pseudo_label": sample.get("pseudo_label", None),
                        "weight":       sample.get("weight", float(sample.get("confidence", 1.0))),
                        # Carry through auxiliary fields used in reward
                        "logic_score":  sample.get("logic_score", 0.0),
                        "confidence":   sample.get("confidence", 0.5),
                    }
                    selected_this_episode.append(kept_sample)

                    # Track embedding for diversity computation
                    if "embedding" in sample and sample["embedding"] is not None:
                        selected_embeddings.append(
                            np.asarray(sample["embedding"], dtype=np.float32)
                        )
                    else:
                        conf    = float(sample.get("confidence", 0.5))
                        entropy = float(sample.get("entropy", 0.0))
                        selected_embeddings.append(
                            np.array([conf, entropy], dtype=np.float32)
                        )

                # Rewards are sparse: only non-zero at episode end
                reward = 0.0
                done   = is_last_step

                if done:
                    reward, episode_new_f1 = self._compute_reward(
                        selected_this_episode, baseline_f1, val_f1_fn
                    )
                    episode_reward = reward

                self.buffer.add(state, action, reward, log_prob, value, done)

            # ---- GAE + PPO update ---------------------------------------
            self.buffer.compute_returns_and_advantages(
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                last_value=0.0,  # episode ended naturally
            )
            self._ppo_update(self.buffer)

            # Reuse the episode F1 already computed inside reward.
            baseline_f1 = episode_new_f1

            if show_progress and hasattr(ep_iter, "set_postfix"):
                ep_iter.set_postfix(
                    selected=len(selected_this_episode),
                    reward=f"{episode_reward:.4f}",
                    f1=f"{baseline_f1:.4f}",
                )

            # Keep the selections from the last episode as the final output
            # For full-pool selection we accumulate across all episodes
            final_kept.extend(selected_this_episode)

        # Return only the essential fields
        return [
            {
                "id":           s["id"],
                "claim":        s["claim"],
                "pseudo_label": s["pseudo_label"],
                "weight":       s["weight"],
                "logic_score":  s.get("logic_score", 0.0),
                "confidence":   s.get("confidence", 0.5),
            }
            for s in final_kept
        ]

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
