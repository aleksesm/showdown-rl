"""Policy / value network for Phase 3.

The Phase 3 baseline architecture: a flat MLP that takes our
observation (features + species embeddings looked up here) and produces
13 action logits plus a scalar state value. Phase 4 will add a pointer-
network variant; that lives in a separate module and reuses the same
`Observation` -> input wiring.

Designed for CleanRL-style single-file implementations: the policy
exposes a `forward(features, species_ids)` that returns `(logits, value)`,
and helpers to sample with action masking. Training loops own their
optimizer and loss; the policy is just a callable nn.Module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .actions import ACTION_DIM
from .state import FEATURES_DIM, SPECIES_IDS_DIM, Observation, num_species

MASK_FILL = -1e9  # large negative instead of -inf so softmax doesn't NaN


class FlatPolicy(nn.Module):
    """Flat 13-action policy with a shared MLP trunk + policy and value heads.

    Architecture:
        species_ids → Embedding(num_species, emb_dim) → flatten →
        cat with features → 2 hidden ReLU layers → {policy_head, value_head}
    """

    def __init__(
        self,
        *,
        feat_dim: int = FEATURES_DIM,
        n_species_slots: int = SPECIES_IDS_DIM,
        species_emb_dim: int = 8,
        hidden: int = 256,
        n_actions: int = ACTION_DIM,
        n_species: int | None = None,
    ):
        super().__init__()
        if n_species is None:
            n_species = num_species()
        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        input_dim = feat_dim + n_species_slots * species_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, features: torch.Tensor, species_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.species_emb(species_ids).flatten(start_dim=-2)
        x = torch.cat([features, emb], dim=-1)
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        obs: Observation,
        mask: np.ndarray,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Sample one action from the masked policy (argmax if `deterministic`). No gradient.

        Returns (action_idx, log_prob_value, state_value). Both scalars are
        Python floats so they can be safely buffered across many turns
        without holding the autograd graph (re-computed during update).
        """
        features = torch.from_numpy(obs.features).unsqueeze(0).to(device)
        species = torch.from_numpy(obs.species_ids).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        logits, value = self.forward(features, species)
        logits = logits.masked_fill(~mask_t, MASK_FILL)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def evaluate_actions(
        self,
        features: torch.Tensor,      # (B, FEATURES_DIM)
        species_ids: torch.Tensor,   # (B, SPECIES_IDS_DIM)
        masks: torch.Tensor,         # (B, ACTION_DIM) bool
        actions: torch.Tensor,       # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-compute log-probs, entropies, and values for buffered transitions.

        Used during policy-gradient updates so we don't have to keep the
        autograd graph alive across many turns of battle simulation.
        """
        logits, values = self.forward(features, species_ids)
        logits = logits.masked_fill(~masks, MASK_FILL)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, entropies, values
