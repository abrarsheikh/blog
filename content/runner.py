# %%
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List
import torch.nn.functional as F
from typing import Tuple, Any
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from typing import Optional
from tqdm.auto import tqdm, trange
from minari import EpisodeData, MinariDataset
import minari
from gymnasium import Env
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv
import math
import warnings
import pickle

# %%
def get_space_dim(space):
    if isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.Box):
        return space.shape[0]
    elif isinstance(space, spaces.Dict):
        return sum([get_space_dim(v) for v in space.values()])
    else:
        raise ValueError("Unsupported observation space")


dataset_ref = "D4RL/antmaze/medium-play-v1"
env_name = "AntMaze_Medium-v4"

base_m_dataset = minari.load_dataset(dataset_ref, download=True)
wrapped_env = base_m_dataset.recover_environment(render_mode="rgb_array")
env = wrapped_env.unwrapped
env.name = env_name

# Environment parameters
observation_dim = get_space_dim(env.observation_space)
action_dim = get_space_dim(env.action_space)
reward_dim = 1
value_dim = 1
transition_dim = observation_dim + action_dim + reward_dim + value_dim

print(f"Observation dim: {observation_dim}, Action dim: {action_dim}")
print(f"Reward dim: {reward_dim}, Value dim: {value_dim}")
print(f"Transition dim: {transition_dim}")

local = not torch.cuda.is_available()

# Model parameters
n_transitions = 10
seq_len = n_transitions * transition_dim
vocab_size = 100
max_bins = vocab_size
discount_factor = 0.99
embedding_dim = 128 if not local else 96
n_heads = 4 if not local else 4
n_blocks = 4 if not local else 4
n_epochs = 50 if not local else 300
batch_size = 256 if not local else 128
use_sep_heads = True
lr = 0.0006
weight_decay = 0.1
betas = (0.9, 0.95)
clip_grad = 1.0
eval_every = 10 if not local else 50
strategy = "uniform"  # "quantile" or "uniform" for discritization

# other parameters
n_episodes: Optional[int] = 1000 if not local else 50
# create a directory to save the model
base_dir = f"data/{dataset_ref}"
os.makedirs(base_dir, exist_ok=True)
checkpoint_path = f"{base_dir}/"
load_checkpoint = (
    False  # set to False if you want to train from scratch even if a checkpoint exists
)
if n_episodes:
    m_dataset = base_m_dataset.sample_episodes(n_episodes)
else:
    m_dataset = base_m_dataset

print(f"Number of episodes: {len(m_dataset)}")

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# %%
class KBinsDiscretizer:
    """
    This class is responsible for encoding and decoding continuous values into discrete bins.
    Number of bins are fixed for all the features.
    """

    def __init__(self, dataset: np.ndarray, n_bins: int, strategy: str = "ordinal"):
        self.n_bins = n_bins
        self.strategy = strategy

        # bin_edges shape: (n_features, n_bins + 1)
        self.bin_edges = self._find_bin_edges(dataset)
        # bin_centers shape: (n_features, n_bins)
        self.bin_centers = (self.bin_edges[:, :-1] + self.bin_edges[:, 1:]) * 0.5
        self.bin_centers_torch = torch.from_numpy(self.bin_centers).float()

    def _find_bin_edges(self, dataset: np.ndarray):
        # dataset shape: (n_samples, n_features)
        bin_edges = []
        if self.strategy == "uniform":
            # min and max values for each feature, shpae: (n_features,)
            mins, maxs = np.min(dataset, axis=0), np.max(dataset, axis=0)
            # bin_edges shape: (n_features, n_bins + 1)
            bin_edges = np.linspace(mins, maxs, self.n_bins + 1).T
        elif self.strategy == "quantile":
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            # bin_edges shape: (n_features, n_bins + 1)
            bin_edges = np.percentile(dataset, quantiles, axis=0).T
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return bin_edges

    def encode(
        self, X: np.ndarray, subslice: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        # use subslice to encode only a part of the features in the X
        if X.ndim == 1:
            # this is to handle the case where we have a single feature
            X = X[None]
        # data shape: (n_samples, n_features)
        edges = self.bin_edges
        if subslice is not None:
            start, end = subslice
            edges = edges[start:end]

        # Xt represents discretized data, shape: (n_samples, n_features)
        Xt = np.zeros(X.shape, dtype=np.long)

        # See documentation of numpy.isclose for an explanation of ``rtol`` and ``atol``.
        rtol = 1.0e-5
        atol = 1.0e-8

        for jj in range(X.shape[1]):
            # Values which are close to a bin edge are susceptible to numeric
            # instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation.
            eps = atol + rtol * np.abs(X[:, jj])
            # why [1:]? bins = edges - 1, but its unclear why we leave out the first element and not the last
            Xt[:, jj] = np.digitize(X[:, jj] + eps, edges[jj][1:])

        # clip the values to be within the range [0, n_bins - 1]
        np.clip(Xt, 0, self.n_bins - 1, out=Xt)

        return Xt

    def decode(
        self, Xt: np.ndarray, subslice: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        # use subslice to decode only a part of the features in the Xt
        if Xt.ndim == 1:
            # this is to handle the case where we have a single feature
            Xt = Xt[None]
        # data shape: (n_samples, n_features)
        centers = self.bin_centers
        if subslice is not None:
            start, end = subslice
            centers = centers[start:end]

        X = np.zeros(Xt.shape, dtype=np.float64)
        for jj in range(Xt.shape[1]):
            X[:, jj] = centers[jj, np.int_(Xt[:, jj])]

        return X

    def expectation(
        self, probs: np.ndarray, subslice: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        # given the probabilities of each bin, calculate the expectation of the feature values
        # perticularly useful when we have a distribution over the bins, maybe from a model after softmax
        # from logits.
        # probs shape: (n_samples, n_features, n_bins)
        if probs.ndim == 1:
            # this is to handle the case where we have a single feature
            probs = probs[None]
        # probs shape: (batch_size, n_features, n_bins)
        # bin_centers shape: (n_features, n_bins) -> (1, n_features, n_bins)
        if torch.is_tensor(probs):
            bin_centers = self.bin_centers_torch.unsqueeze(0)
        else:
            # bin_centers shape: (n_features, n_bins) -> (1, n_features, n_bins)
            bin_centers = np.expand_dims(self.bin_centers, axis=0)

        if subslice is not None:
            start, end = subslice
            bin_centers = bin_centers[:, start:end]

        # use formula E[X] = sum(p(x) * x) for all x
        # (batch_size, n_features, n_bins) * (1, n_features, n_bins) -> sum (batch_size, n_features, n_bins) -> (batch_size, n_features)
        X = (probs * bin_centers).sum(axis=-1)
        return X

    def to(self, device):
        self.bin_centers_torch = self.bin_centers_torch.to(device)

# %%
# Test array
test_arr = np.array([[1, 2], [3, 4], [5, 6]])

# Initialize the discretizer
discretizer = KBinsDiscretizer(test_arr, 1000, strategy="uniform")

# Encode and decode the test array
encoded = discretizer.encode(test_arr)
decoded = discretizer.decode(encoded)

# Check if the decoded array is close to the original array
assert np.isclose(
    decoded, test_arr, atol=1e-2
).all(), f"Decoded array {decoded} is not close to the original array {test_arr}"

# Generate random probabilities
probs = F.softmax(torch.from_numpy(np.random.rand(3, 2, 1000)), dim=-1).numpy()

# Calculate the expectation
expectation = discretizer.expectation(probs)

# Check if the expectation is close to the mean of the test array
expected_mean = np.tile(np.mean(test_arr, axis=0), (3, 1))
assert np.isclose(
    expectation, expected_mean, atol=1e-1
).all(), f"Expectation {expectation} is not close to the expected mean {expected_mean}"

print("All tests passed successfully.")

# %%
def flatten_space(s_dict: Any, space: spaces.Space) -> np.ndarray:
    if isinstance(space, spaces.Discrete):
        return s_dict
    elif isinstance(space, spaces.Box):
        return s_dict
    elif isinstance(space, spaces.Dict):
        return np.concatenate(
            [flatten_space(s_dict[k], space.spaces[k]) for k in space.spaces.keys()],
            axis=-1,
        )
    else:
        raise ValueError("Unsupported observation space")


def unflatten_space(s_flat: np.ndarray, space: spaces.Space) -> dict:
    if isinstance(space, spaces.Discrete):
        return s_flat
    elif isinstance(space, spaces.Box):
        return s_flat
    elif isinstance(space, spaces.Dict):
        s_dict = {}
        start = 0
        for k, v in space.spaces.items():
            end = start + get_space_dim(v)
            s_dict[k] = unflatten_space(s_flat[:, start:end], v)
            start = end
        return s_dict
    else:
        raise ValueError("Unsupported observation space")


# Test the flatten_space_dict and unflatten_space_dict functions
test_dict = {"obs": np.array([[1, 2, 3], [4, 5, 6]]), "act": np.array([[0], [1]])}
test_space = spaces.Dict(
    {"obs": spaces.Box(low=0, high=10, shape=(3,)), "act": spaces.Discrete(2)}
)
test_flat = flatten_space(test_dict, test_space)
test_unflat = unflatten_space(test_flat, test_space)

assert np.isclose(
    test_flat, np.array([[0, 1, 2, 3], [1, 4, 5, 6]])
).all(), f"Flattened array {test_flat} is not as expected."
assert np.isclose(
    test_unflat["obs"], test_dict["obs"]
).all(), f"Unflattened observation {test_unflat['obs']} is not as expected."
assert np.isclose(
    test_unflat["act"], test_dict["act"]
).all(), f"Unflattened action {test_unflat['act']} is not as expected."

# test discrete space
test_dict = np.array([[0], [1]])
test_space = spaces.Discrete(2)
test_flat = flatten_space(test_dict, test_space)
test_unflat = unflatten_space(test_flat, test_space)

assert np.isclose(
    test_flat, test_dict
).all(), f"Flattened array {test_flat} is not as expected."
assert np.isclose(
    test_unflat, test_dict
).all(), f"Unflattened array {test_unflat} is not as expected."

# test box space
test_dict = np.array([[1, 2, 3], [4, 5, 6]])
test_space = spaces.Box(low=0, high=10, shape=(3,))
test_flat = flatten_space(test_dict, test_space)
test_unflat = unflatten_space(test_flat, test_space)

assert np.isclose(
    test_flat, test_dict
).all(), f"Flattened array {test_flat} is not as expected."
assert np.isclose(
    test_unflat, test_dict
).all(), f"Unflattened array {test_unflat} is not as expected."

print("All tests passed successfully.")

# %%
def join_trajectory(env: Env, episode: EpisodeData, discount: float = 0.99):
    # Convert the object of type EpisodeData to a numpy array. EpisodeData
    # contains the following fields: observations, actions, rewards, other
    # and each of these fields is a numpy array. We need to concatenate
    # these arrays along the last axis to get a single array for each time.

    success = episode.infos['success']
    # end of the trajectory is the first success or the end of the episode
    success_indices = np.where(success)[0]
    if len(success_indices) > 0:
        last_success_idx = success_indices[0]
        trajectory_len = last_success_idx + 1
    else:
        last_success_idx = -1
        trajectory_len = len(episode.rewards)
    # shape (trajectory_len, observation_dim)
    observations = episode.observations
    # shape (trajectory_len, action_dim)
    actions = episode.actions
    # shape (trajectory_len, action_dim)
    rewards = episode.rewards[:trajectory_len]

    # use values to store the rewards to go
    # for a given time step, the value is the sum of rewards from that time step
    # to the end of the trajectory, discounted by discount factor at each time step
    values = np.zeros_like(rewards, dtype=np.float32)
    # calculate discounts for each time step
    discounts = discount ** np.arange(trajectory_len)
    # calculate rewards to go with discount
    for t in range(trajectory_len):
        values[t] = (rewards[t + 1 :].T * discounts[: -t - 1]).sum()

    # drop the last state because we don't have a reward for it
    states = flatten_space(observations, env.observation_space)
    states = states[:trajectory_len]
    actions = flatten_space(actions, env.action_space)
    actions = actions[:trajectory_len]
    rewards = rewards[:, None]
    values = values[:, None]

    # shape (trajectory_len, observation_dim + action_dim + reward_dim + value_dim)
    joined = np.concatenate([states, actions, rewards, values], axis=-1)

    return joined


class DiscretizeDataset(Dataset):
    # Each input into the sequence model needs to be (batch_size, tokens)
    # output should be in groups of transitions
    def __init__(
        self,
        env: Env,
        m_dataset: MinariDataset,
        n_transitions: int,
        discount: float = 0.99,
        max_bins: int = 1000,
        strategy: str = "quantile",
        cache_path: Optional[str] = None,
        load_checkpoint: bool = True,
    ):
        self.m_dataset = m_dataset
        self.n_transitions = n_transitions

        ds_len = len(self.m_dataset)

        self.cache_name = (
            f"joined_trajectories_{n_transitions}_{max_bins}_{strategy}_{ds_len}.pkl"
        )
        cache_path = os.path.join(cache_path, self.cache_name) if cache_path else None

        if load_checkpoint and cache_path is not None and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, "rb") as f:
                self.joined_trajectories = pickle.load(f)
        else:
            # this list will contain the joined trajectories, each item in the list
            # is a trajectory of shape (trajectory_len, observation_dim + action_dim + reward_dim + value_dim)
            # and that trajectory is one episodedata from the m_dataset
            self.joined_trajectories = []
            for episode in m_dataset:
                self.joined_trajectories.append(join_trajectory(env, episode, discount))

            print(f"Caching dataset to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(self.joined_trajectories, f)

        self.discretizer = KBinsDiscretizer(
            n_bins=max_bins,
            strategy=strategy,
            # concatenate all the trajectories
            # shape (n_samples * trajectory_len, observation_dim + action_dim + reward_dim + value_dim)
            dataset=np.concatenate(self.joined_trajectories, axis=0),
        )

        # we need a dataset for training sequence model
        # given that we need a sequence of n_transitions, we need to generate
        # indices such that we can get n_transitions from each trajectory
        indices = []
        for traj_idx, joined_trajectory in enumerate(self.joined_trajectories):
            traj_len = joined_trajectory.shape[0]
            end = traj_len - 1
            for i in range(end):
                indices.append((traj_idx, i, i + n_transitions))

        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start, end = self.indices[idx]
        # sample a sequence of n_transitions from trajectory at traj_idx
        joined = self.joined_trajectories[traj_idx][start:end]
        loss_pad_mask = np.ones((self.n_transitions, joined.shape[-1]), dtype=np.long)
        # some sequences may be shorter than n_transitions, pad them with zeros
        # and set the mask to zero for the padded part, this mask will be used
        # to mask the loss when calculating the loss
        if joined.shape[0] < self.n_transitions:
            # pad along dimension 0, zero padding at the beginning
            # and (self.n_transitions - joined.shape[0]) padding at the end
            joined = np.pad(
                joined,
                ((0, self.n_transitions - joined.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            loss_pad_mask[joined.shape[0] :] = 0

        # since transformer model expects discrete values, we need to encode the
        # continuous values into discrete bins
        # shape (n_transitions, transition_dim) -> (n_transitions, transition_dim)
        joined_discretized = self.discretizer.encode(joined)
        # shape (n_transitions, transition_dim) -> (n_transitions * transition_dim)
        # i'e [s1, a1, r1, v1, s2, a2, r2, v2, ...]
        joined_discretized = joined_discretized.reshape(-1).astype(np.long)
        loss_pad_mask = loss_pad_mask.reshape(-1)
        # return input, target, and mask
        # since sequence model predicts the next token, target is the next token in the sequence
        return joined_discretized[:-1], joined_discretized[1:], loss_pad_mask[:-1]

# %%
dataset = DiscretizeDataset(
    env=env,
    m_dataset=m_dataset,
    n_transitions=n_transitions,
    discount=discount_factor,
    max_bins=max_bins,
    strategy=strategy,
    cache_path=checkpoint_path,
    load_checkpoint=load_checkpoint,
)

print(f"Length of dataset: {len(dataset)}")
print(f"Shape of dataset: {dataset[0][0].shape}")

# %%
def weight_decay_groups(
    model, whitelist_modules, blacklist_modules, blacklist_named=None
):
    # from https://github.com/karpathy/minGPT
    decay, no_decay = set(), set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            # starts with for rnn's, endswith other
            if pn.startswith("bias") or pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.startswith("weight") or pn.endswith("weight")) and isinstance(
                m, blacklist_modules
            ):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif (pn.startswith("weight") or pn.endswith("weight")) and isinstance(
                m, whitelist_modules
            ):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)

    if blacklist_named is not None:
        for name in blacklist_named:
            no_decay.add(name)  # also no decay

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    if len(inter_params) != 0:
        warnings.warn(
            f"parameters {str(inter_params)} made it into both decay/no_decay sets! They will be added to only no_decay by default."
        )
        decay = decay - no_decay

    inter_params = decay & no_decay
    union_params = decay | no_decay
    if len(param_dict.keys() - union_params) != 0:
        warnings.warn(
            f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set! They will be added to decay by default."
        )
        decay = decay | (param_dict.keys() - union_params)

    optim_groups = {
        "decay": [param_dict[pn] for pn in sorted(list(decay))],
        "nodecay": [param_dict[pn] for pn in sorted(list(no_decay))],
    }
    return optim_groups


def round_to_multiple(number, multiple):
    """
    Rounds a given number up to the nearest multiple of a specified value.

    Args:
        number (int or float): The number to be rounded.
        multiple (int or float): The multiple to which the number should be rounded.

    Returns:
        int or float: The number rounded up to the nearest multiple of the specified value.
    """
    pad = (multiple - number % multiple) % multiple
    return number + pad


# Test the round_to_multiple function
assert round_to_multiple(5, 3) == 6
assert round_to_multiple(6, 3) == 6
assert round_to_multiple(7, 3) == 9


class GPTScheduler:
    """
    Linear warmup to optimizer inital_lr for #warmup_tokens,
    then cosine decay to inital_lr * final_lr_ratio for the rest #final_tokens
    source: https://github.com/karpathy/minGPT
    """

    def __init__(
        self, optimizer, warmup_tokens, final_tokens, final_lr_ratio=0.1, decay=True
    ):
        self.optimizer = optimizer
        # assuming that lr same for all group
        self.init_lr = optimizer.param_groups[0]["lr"]

        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.final_lr_ratio = final_lr_ratio
        self.decay = decay

        self.tokens_count = 0.0

    def step(self, batch_size):
        lr_mult = self.__get_lr_multiplier(batch_size)

        for group in self.optimizer.param_groups:
            group["lr"] = self.init_lr * lr_mult

    def get_current_lr(self):
        lr_mult = self.__get_lr_multiplier(0.0)
        return self.init_lr * lr_mult

    def __get_lr_multiplier(self, batch_size):
        self.tokens_count += batch_size

        assert (
            self.tokens_count <= self.final_tokens
        ), f"number of tokens {self.tokens_count} already bigger than number of tokens for one cycle"

        if self.tokens_count < self.warmup_tokens:
            lr_mult = float(self.tokens_count) / float(max(1, self.warmup_tokens))
        elif self.tokens_count >= self.warmup_tokens and self.decay:
            tokens_passed = self.tokens_count - self.warmup_tokens
            tokens_left = self.final_tokens - self.warmup_tokens

            progress = float(tokens_passed) / float(max(1, tokens_left))
            lr_mult = max(
                self.final_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress))
            )
        else:
            lr_mult = 1.0

        return lr_mult

    def state_dict(self):
        # just for checkpoint callback
        pass

# %%
class Block(nn.Module):
    # Transformer block
    def __init__(
        self,
        seq_len,
        embedding_dim: int,
        transition_dim: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embedding_dim, n_heads, batch_first=True, dropout=attention_dropout
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.fc_norm = nn.LayerNorm(embedding_dim)

        self.drop = nn.Dropout(residual_dropout)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        self.seq_len = seq_len

        # mask value of true means that the value is not allowed to be attended to
        mask = ~torch.tril(torch.ones(seq_len, seq_len)).bool()
        # transition_dim - 1 stores rewards to go, we don't want to attend to them because they contain future information
        mask[:, transition_dim - 1 :: transition_dim] = True
        self.register_buffer("mask", mask)

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x shape (batch_size, n_tokens, embedding_dim) in prefill mode else (batch_size, 1, embedding_dim)
        # kv_cache shape (batch_size, n_tokens, embedding_dim) in inference mode else None
        _, n_tokens, _ = x.shape

        # normalize the input before passing it to the attention layer
        x_norm = self.attn_norm(x)

        if kv_cache is None:
            # when kv_cache is None, we are in prefill mode

            # attn_mask shape (seq_len, seq_len), but incoming shape is (batch_size, n_tokens, embedding_dim)
            # so filter the mask to the correct size (n_tokens, n_tokens)
            attn_mask = self.mask[:n_tokens, :n_tokens]
            q, k, v = x_norm, x_norm, x_norm
        else:
            assert n_tokens == 1, "kv_cache can only be None with a single token"
            # +1 because we are adding a new token
            assert kv_cache.shape[1] + 1 <= self.seq_len, "kv_cache is too large"

            # attn_mask is None because we are running in inference mode, processing one token at a time
            # and this token is not allowed to attend to future tokens
            attn_mask = None
            q, k, v = (
                x_norm,
                # shape (batch_size, n_tokens + 1, embedding_dim)
                torch.cat([kv_cache, x_norm], dim=1),
                torch.cat([kv_cache, x_norm], dim=1),
            )

        new_kv_cache = k

        # x shape (batch_size, n_tokens, embedding_dim) in prefill mode else (batch_size, 1, embedding_dim)
        x = x + self.drop(
            self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
        )

        x = x + self.mlp(self.fc_norm(x))

        return x, new_kv_cache


class EinLinear(nn.Module):
    def __init__(
        self, n_models: int, in_features: int, out_features: int, bias: bool = True
    ):
        super().__init__()
        self.n_models = n_models
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, model_idx: Optional[int] = None) -> torch.Tensor:
        if model_idx is None:
            # when model_idx is None, we are in prefill mode
            # (n_models, out_features, in_features) * (batch_size, n_models, in_features) -> (batch_size, n_models, out_features)
            output = torch.einsum("eoi,bei->beo", self.weight, x)
        else:
            # when model_idx is not None, we are in inference mode
            # shape (batch_size, in_features) * (out_features, in_features).T -> (batch_size, out_features)
            output = x @ self.weight[model_idx].T

        if self.bias is not None:
            raise RuntimeError()

        return output


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        n_heads: int,
        transition_dim: int,
        n_blocks: int,
        vocab_size: int,
        dropout_embedding: float = 0.05,
        attention_dropout: float = 0.05,
        residual_dropout: float = 0.05,
        use_sep_heads: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.transition_dim = transition_dim
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size

        # our input contains transition_dim types of tokens and each token is from a vocab of size vocab_size
        # so the total number of tokens is transition_dim * vocab_size
        self.token_embedding = nn.Embedding(
            vocab_size * transition_dim, self.embedding_dim
        )
        # learnable positional embedding
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, seq_len, self.embedding_dim)
        )

        self.dropout_embedding = nn.Dropout(dropout_embedding)

        # create n_blocks of transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.seq_len,
                    self.embedding_dim,
                    self.transition_dim,
                    self.n_heads,
                    attention_dropout,
                    residual_dropout,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.norm = nn.LayerNorm(self.embedding_dim)

        self.use_sep_heads = use_sep_heads

        if not self.use_sep_heads:
            # project the output of the transformer to the vocab size
            # since each token type is from a vocab of size vocab_size
            # we can do this. But for instance if every token type used different
            # number of bins, then we would have handled this differently. But dont worry
            # that is not the case here.
            self.fc = nn.Linear(self.embedding_dim, vocab_size)
        else:
            self.fc = EinLinear(
                self.transition_dim, self.embedding_dim, vocab_size, bias=False
            )

        # self.apply is a crazy function that applies the given function recursively to every submodule
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # standard practice in transformer models
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TrajectoryTransformer):
            torch.nn.init.normal_(module.positional_embedding, mean=0.0, std=0.02)

    def get_seq_len(self):
        return self.seq_len

    def _pad_to_full_transition(self, tokens: torch.Tensor) -> torch.Tensor:
        # pad the tokens to full transition_dim
        batch_size, n_tokens, _ = tokens.shape
        n_pad = round_to_multiple(n_tokens, self.transition_dim) - n_tokens
        padding = torch.zeros(
            batch_size, n_pad, self.embedding_dim, device=tokens.device
        )
        x_pad = torch.cat([tokens, padding], dim=1)
        return x_pad, n_pad

    def _offset_tokens(
        self, tokens: torch.Tensor, kv_caches: Optional[List] = None
    ) -> torch.Tensor:
        # for beginners, this function may be a bit confusing. So let me explain

        # our input consists of transition_dim types of tokens
        # and each token is from a vocab of size vocab_size. So total
        # there are transition_dim * vocab_size unique tokens. In contrast
        # to NLP where we have just one token type(the word) and each word
        # is from a vocab of size vocab_size (50k in llama).
        # So to bridge this gap, we need to project each token's local vocab
        # into the global vocab space. And the way we do this is by offsetting
        # each token type by a factor of vocab_size.
        # eg. if we have 3 token types and vocab_size is 10, then the tokens
        # will be offset by [0, 10, 20] respectively.
        # given input [2, 6, 3, 1, 2, 5] and vocab_size 10,
        # the output will be [2, 16, 23, 11, 12, 25]

        n_tokens = tokens.shape[1] if kv_caches is None else kv_caches[0].shape[1] + 1
        # calculate the number of transitions in the input
        n_transition = int(np.ceil(n_tokens / self.transition_dim))

        # if transition_dim is 4, and vocab_size is 10, then the offsets will be
        # [0, 10, 20, 30]
        # shape (transition_dim,)
        offsets = (
            torch.arange(self.transition_dim, device=tokens.device) * self.vocab_size
        )
        # repeat the offset n_transition times
        # shape (n_transition * transition_dim,)
        offsets = offsets.repeat(n_transition)
        if kv_caches is not None:
            # in inference mode, we need to offset the last token only
            offset_idx = offsets[:n_tokens][-1] + tokens
        else:
            # add the offsets to the tokens, and truncate the tokens to n_tokens
            offset_idx = offsets[:n_tokens] + tokens
        return offset_idx

    def forward(
        self, tokens: torch.Tensor, kv_caches: Optional[List] = None
    ) -> torch.Tensor:
        # tokens shape (batch_size, n_tokens) in prefill mode else (batch_size, 1)
        batch_size, n_tokens = tokens.shape
        assert (
            n_tokens <= self.seq_len
        ), f"n_tokens {n_tokens} is greater than seq_len {self.seq_len}"

        if kv_caches is not None:
            assert n_tokens == 1, "kv_caches can only be used with a single token"

        # project each token into their vocab space, this is similar to tokenization
        # in NLP where we project each word into their vocab space
        # (batch_size, n_tokens)
        offset_idx = self._offset_tokens(tokens, kv_caches)

        # (batch_size, n_tokens) -> (batch_size, n_tokens, embedding_dim)
        tokens = self.token_embedding(offset_idx)

        if kv_caches is not None:
            # in inference mode
            idx = kv_caches[0].shape[1]
            # (1, 1, embedding_dim)
            positional_embedding = self.positional_embedding[:, idx : idx + 1]
        else:
            # in prefill mode
            # initialize kv_caches to None
            kv_caches = [None for _ in range(self.n_blocks)]
            # (1, n_tokens, embedding_dim)
            positional_embedding = self.positional_embedding[:, :n_tokens]

        # (batch_size, n_tokens, embedding_dim) -> (batch_size, n_tokens, embedding_dim)
        tokens = self.dropout_embedding(tokens + positional_embedding)

        new_kv_caches = []
        for block, kv_cache in zip(self.blocks, kv_caches):
            tokens, new_kv_cache = block(tokens, kv_cache)
            new_kv_caches.append(new_kv_cache)

        # (batch_size, n_tokens, embedding_dim) -> (batch_size, n_tokens, embedding_dim)
        tokens = self.norm(tokens)

        if self.use_sep_heads:
            # by using separate heads, we can route each token type to a different module
            # this can be useful when each token type uses different number of bins or
            # we want to give more capacity for the model to learn.
            if kv_caches[0] is None:
                # in prefill mode, we need to calculate the logits for each token type
                # (batch_size, n_tokens, embedding_dim) -> (batch_size, n_tokens + n_pad, vocab_size)
                x_pad, n_pad = self._pad_to_full_transition(tokens)
                # (batch_size, n_tokens + n_pad, vocab_size) -> (batch_size * n_transitions, transition_dim, embedding_dim)
                x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
                # (batch_size * n_transitions, transition_dim, embedding_dim) -> (batch_size * n_transitions, transition_dim, vocab_size)
                logits = self.fc(x_pad, model_idx=None)
                # (batch_size * n_transitions, transition_dim, vocab_size) -> (batch_size, n_tokens + n_pad, vocab_size)
                logits = logits.reshape(batch_size, n_tokens + n_pad, self.vocab_size)
                # truncate the logits to n_tokens
                logits = logits[:, :n_tokens, :]
            else:
                # in inference mode, we need to calculate the logits for the last token type
                # infer the model index to route the token to the correct model.
                cache_size = kv_cache[0].shape[1]
                model_idx = cache_size % self.transition_dim
                # (batch_size, 1, embedding_dim) -> (batch_size, embedding_dim) -> (batch_size, vocab_size) -> (batch_size, 1, vocab_size)
                logits = self.fc(tokens.squeeze(1), model_idx).unsqueeze(1)
        else:
            # (batch_size, n_tokens, embedding_dim) -> (batch_size, n_tokens, vocab_size)
            logits = self.fc(tokens)
        return logits, new_kv_caches

# %%
def sample_token_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    This function return exactly one token from the logits.
    We have options to sample from the logits using
    1. Greedy sampling
    2. Top-k sampling
    3. Temperature scaling

    """
    # logits shape (batch_size, vocab_size) representing the logits of the next token

    # Apply temperature scaling, the higher the temperature, the more uniform the distribution
    # the lower the temperature, the more peaked the distribution
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None:
        # Apply top-k sampling
        # (batch_size, vocab_size) -> (batch_size, top_k)
        v, indices = torch.topk(logits, top_k, dim=-1)

        # Next instruction is a bit tricky, but it simply selects the top-k tokens
        # set all logits to -inf except the top-k indices
        # v[:, [-1]] might be a bit confusing, but it simply selects the last element
        # along dim=1, and the result is a tensor of shape (batch_size, 1)
        logits[logits < v[:, [-1]]] = -float("Inf")
    # Calculate the probabilities from the logits
    probs = F.softmax(logits, dim=-1)
    if not greedy:
        # Sample from the top-k indices
        # (batch_size, top_k) -> (batch_size, 1)
        idx = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy sampling
        _, idx = torch.max(probs, dim=-1)
    return idx


def sample_tokens(
    model: nn.Module,
    context: nn.Module,
    kv_caches: Optional[List],
    n_steps: int,
    temperature: float = 1.0,
    greedy: bool = False,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a sequence of tokens from the model.

    Args:
        model (nn.Module): The model to sample from.
        context (nn.Module): The context to condition the sampling on.
            shape (batch_size, n_tokens).
        n_steps (int): The number of steps to sample.
        temperature (float): The temperature scaling factor.
        greedy (bool): Whether to sample greedily.
        top_k (Optional[int]): The top-k sampling parameter.

    Returns:
        torch.Tensor: The sampled tokens.
    """
    # tensor to store the logits of the next   sampled tokens
    raw_logits = torch.zeros(
        context.shape[0], n_steps, vocab_size, device=context.device
    )
    if kv_caches is None:
        # when kv_caches is None, we are in prefilling step
        logits, kv_caches = model(context, kv_caches)
        # Sample the next token
        # (batch_size, 1)
        token = sample_token_from_logits(
            logits[:, -1], temperature=temperature, greedy=greedy, top_k=top_k
        )

        context = torch.cat([context, token], dim=-1)

        raw_logits[:, 0] = logits[:, -1]
        # since we already did one step, we need to sample n_steps - 1
        steps = range(1, n_steps)
    else:
        steps = range(n_steps)

    for i in steps:
        # crop the context so that it doesn't exceed the seq_len
        curr_context_len = context.shape[1]
        n_crop = round_to_multiple(
            max(0, curr_context_len - model.get_seq_len()), transition_dim
        )
        if n_crop > 0:
            # since we are cropping from the left, we need to update the kv_caches
            kv_caches = [kv[:, n_crop:] for kv in kv_caches]
        # Get the model's prediction
        # (batch_size, 1) -> (batch_size, 1, vocab_size)
        logits, kv_caches = model(context[:, -1:], kv_caches)
        # Sample the next token
        # (batch_size, 1)
        token = sample_token_from_logits(
            logits[:, -1], temperature=temperature, greedy=greedy, top_k=top_k
        )

        context = torch.cat([context, token], dim=-1)

        raw_logits[:, i] = logits[:, -1]
    return context, kv_caches, raw_logits

# %%
# This functions are probably the most important functions in this notebook and
# also the most complex.


def vec_beam_plan(
    model: nn.Module,
    discretizer: KBinsDiscretizer,
    context: torch.Tensor,
    beam_width: int,
    beam_steps: int,
    beam_context: int,
    sample_expansion: int,
    observation_dim: int,
    action_dim: int,
    reward_dim: int,
    value_dim: int,
    transition_dim: int,
    obs_top_k: Optional[int] = None,
    act_top_k: Optional[int] = None,
    rew_top_k: Optional[int] = None,
    temperature: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """
    In the most simplest terms, this function is responsible for planning a sequence of actions
    that maximizes the expected rewards conditioned on the context.

    It uses beam search to explore the space of possible plans. Beam search is a heuristic search
    algorithm that explores a graph by expanding the most promising nodes in a limited set called the beam.

    The concept of beam search is simple, but the implementation can be a bit tricky mainly because
    we are processing multiple sequences in parallel. This is where the complexity comes from.
    """
    batch_size = context.shape[0]
    tokens_context_size = beam_context * transition_dim
    n_crop = round_to_multiple(
        max(0, context.shape[1] - tokens_context_size), transition_dim
    )
    context = context[:, n_crop:]
    # context shape (batch_size, seq_len) -> (beam_width, beam_width, seq_len)
    plan = context.unsqueeze(1).repeat(1, beam_width, 1)

    # tensor to store the rewards obtained from environment
    # the +1 is non-intuitive, but it is because we need to store the value at t+1
    # you will see this later.
    rewards = torch.zeros(batch_size, beam_width, beam_steps + 1, device=context.device)
    discounts = discount_factor ** torch.arange(beam_steps + 1, device=context.device)

    # because beam plan start with a fresh context, we need to prefill the model
    # first with the context, hence kv_caches is None
    kv_caches = None
    for t in trange(beam_steps, desc="Beam Search", leave=False):
        # sample_expansion is not strictly necessary, but it is used to increase the number of samples
        #   which should allow us to explore more diverse plans. The reason this works is because the way
        #   we sample tokens is stochastic, so by sampling more tokens, we are able to explore more diverse plans.
        # (batch_size, beam_width, n_tokens) -> (batch_size, beam_width * sample_expansion, n_tokens)
        #   -> (batch_size * beam_width * sample_expansion, n_tokens)
        plan = plan.repeat(1, sample_expansion, 1).flatten(0, 1)
        # (batch_size, beam_width, beam_steps + 1) -> (batch_size * beam_width * sample_expansion, beam_steps + 1)
        rewards = rewards.repeat(1, sample_expansion, 1).flatten(0, 1)

        if kv_caches is not None:
            # When we are in inference mode, we need to expand the kv_caches
            # (batch_size * beam_width, n_tokens, embedding_dim) -> (batch_size * beam_width * sample_expansion, n_tokens, embedding_dim)
            new_kv_caches = []
            for kv in kv_caches:
                _, n_tokens, embedding_dim = kv.shape
                new_kv_cache = (
                    kv.view(batch_size, beam_width, n_tokens, embedding_dim)
                    .repeat(1, sample_expansion, 1, 1)
                    .flatten(0, 1)
                )
                new_kv_caches.append(new_kv_cache)
            kv_caches = new_kv_caches

        # sample actions
        # plan (batch_size * beam_width * sample_expansion, n_tokens) -> (batch_size * beam_width * sample_expansion, n_tokens + action_dim)
        # kv_caches is updated with the new action tokens
        plan, kv_caches, _ = sample_tokens(
            model,
            plan,
            kv_caches,
            n_steps=action_dim,
            top_k=act_top_k,
            temperature=temperature,
            greedy=greedy,
        )

        # sample rewards and values
        # plan (batch_size * beam_width * sample_expansion, n_tokens) -> (batch_size * beam_width * sample_expansion, n_tokens + reward_dim + value_dim)
        # kv_caches is updated with the new reward and value tokens
        # logits shape (batch_size * beam_width * sample_expansion, reward_dim + value_dim, vocab_size)
        plan, kv_caches, logits = sample_tokens(
            model,
            plan,
            kv_caches,
            n_steps=reward_dim + value_dim,
            top_k=rew_top_k,
            temperature=temperature,
            greedy=greedy,
        )

        # calculate probabilities from logits
        probs = F.softmax(logits, dim=-1)

        # calculate the expected rewards and values
        # (batch_size * beam_width * sample_expansion, reward_dim + value_dim, vocab_size)
        #   -> (batch_size * beam_width * sample_expansion, reward_dim + value_dim)
        rewards_and_values = discretizer.expectation(
            probs, subslice=(transition_dim - reward_dim - value_dim, transition_dim)
        )

        rewards[..., t : t + reward_dim + value_dim] = rewards_and_values
        # Did you notice that rewards contains rewards at t and values at t+1, why?
        #   This is only a trick to make it easier to calculate the value at t. In the next step, the value at t+1
        #   will be overwritten by the actual reward at t+1.

        # Let's talk about how we calculate the value, the value here represents the rewards to go starting beginning of beam plan.
        #   when we want to calculate value (rewards to go) at t, we need to consider discounted rewards from 0 to t
        #   and also future discounted rewards from t+1 to end.
        # (batch_size * beam_width * sample_expansion, beam_steps + 1) * (beam_steps + 1) -> (batch_size * beam_width * sample_expansion)
        # the reason we care of values is that it helps us to select the best plans
        values = (rewards * discounts).sum(dim=-1)

        # select the top-k values
        values, idx = torch.topk(values.view(batch_size, -1), k=beam_width, dim=-1)
        # (batch_size, beam_width) -> (batch_size, beam_width, 1)
        idx = idx.unsqueeze(-1)

        # shape (batch_size * beam_width * sample_expansion, beam_steps + 1) -> (batch_size, beam_width * sample_expansion, beam_steps + 1)
        rewards = rewards.view(batch_size, beam_width * sample_expansion, -1)

        # the gather operation is a bit tricky, but it is used to select the rewards corresponding to the top-k values
        # for every batch, select the rewards corresponding to the top-k values
        # since idx contains the indices along the beam_width * sample_expansion dimension
        # we need to repeat the idx along the last dimension to match the rewards shape,
        # and then use it to select the rewards
        # (batch_size, beam_width * sample_expansion, beam_steps + 1) -> (batch_size, beam_width, beam_steps + 1)
        rewards = torch.gather(rewards, 1, idx.repeat(1, 1, beam_steps + 1))

        # select the top-k plans
        # shape (batch_size * beam_width * sample_expansion, n_tokens) -> (batch_size, beam_width * sample_expansion, n_tokens)
        plan = plan.view(batch_size, beam_width * sample_expansion, -1)
        # shape (batch_size, beam_width * sample_expansion, n_tokens) -> (batch_size, beam_width, n_tokens)
        plan = torch.gather(plan, 1, idx.repeat(1, 1, plan.shape[-1]))

        # select the top-k kv_caches
        best_kv_caches = []
        for kv in kv_caches:
            _, n_tokens, embedding_dim = kv.shape
            kv = kv.view(
                batch_size, beam_width * sample_expansion, n_tokens, embedding_dim
            )
            # same idea as above, repeat idx along the last 2 dimensions
            # kv shape (batch_size, beam_width * sample_expansion, n_tokens, embedding_dim) -> (batch_size, beam_width, n_tokens, embedding_dim)
            kv = torch.gather(
                kv, 1, idx.unsqueeze(-1).repeat(1, 1, n_tokens, embedding_dim)
            )
            best_kv_caches.append(kv.flatten(0, 1))

        if t < beam_steps - 1:
            # sample observations only if we are not at the last step, why?
            # because beam plan has to end with a valid transition [...., obs, act, rew, val]

            # plan (batch_size, beam_width, n_tokens) -> (batch_size * beam_width, n_tokens)
            plan = plan.view(batch_size * beam_width, -1)

            # sample observations
            # plan (batch_size * beam_width, n_tokens) -> (batch_size * beam_width, n_tokens + observation_dim)
            plan, kv_caches, _ = sample_tokens(
                model,
                plan,
                best_kv_caches,
                n_steps=observation_dim,
                top_k=obs_top_k,
                temperature=temperature,
                greedy=greedy,
            )
            # plan (batch_size * beam_width, n_tokens + observation_dim) -> (batch_size, beam_width, n_tokens + observation_dim)
            plan = plan.view(batch_size, beam_width, -1)

    # (batch_size, beam_width) -> (batch_size)
    # for each batch, select the plan with the highest value and return it's index
    argmax = torch.argmax(values, dim=-1)

    # select the best plan
    # (batch_size, beam_width, n_tokens) -> (batch_size, n_tokens)
    best_plan = plan[torch.arange(batch_size), argmax]
    # filter out the context tokens and return the best plan as obtained from the beam search
    best_plan = best_plan[:, context.shape[1] :]
    return best_plan


@torch.no_grad()
def vec_rollout(
    model: nn.Module,
    env: DummyVecEnv,
    discretizer: KBinsDiscretizer,
    beam_width: int,
    beam_steps: int,
    beam_context: int,
    sample_expansion: int,
    observation_dim: int,
    action_dim: int,
    reward_dim: int,
    value_dim: int,
    transition_dim: int,
    max_steps: int,
    plan_every: int,
    obs_top_k: Optional[int] = None,
    act_top_k: Optional[int] = None,
    rew_top_k: Optional[int] = None,
    temperature: float = 1.0,
    greedy: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """
    What is a rollout? A rollout is a simulation of an agent interacting with the environment
    by following a plan. The plan is generated by the model using beam search. The model
    predicts the next action, reward, and observation conditioned on the context. The context
    is the history of the agent's interaction with the environment.

    This function is responsible for performing a rollout using the model and the environment
    and returning the total rewards obtained by the agent.

    Similar to the vec_beam_plan function, this function is a bit complex because it processes
    multiple sequences in parallel. The complexity comes from the fact that we are using a vectorized
    environment, which means that we are processing multiple environments in parallel.
    """
    assert (
        plan_every <= beam_steps
    ), f"plan_every {plan_every} should be less than or equal to beam_steps {beam_steps}"

    # reset the environment amd get the initial observation
    # in most environments, the initial observation selected randomly.
    obs = env.reset()

    # obs shape (num_envs, observation_dim)
    obs = flatten_space(obs, env.observation_space)
    total_rewards = np.zeros(env.num_envs)
    context = torch.zeros(
        (env.num_envs, (max_steps + 1) * transition_dim),
        device=device,
        dtype=torch.long,
    )
    # context_idx is used to keep track of the current index in the context
    context_idx = 0

    # discretize the observation
    # obs_token shape (num_envs, observation_dim)
    obs_token = discretizer.encode(obs, subslice=(0, observation_dim))

    value_placeholder = np.ones((env.num_envs, value_dim)) * 1e6

    # update the context with the initial observation
    context[:, :observation_dim] = torch.tensor(obs_token, device=device)

    # tensor to keep track of which environments are done
    dones = np.zeros(env.num_envs, dtype=np.bool)

    # usually max_steps is set to default max_num_steps in the environment
    for t in trange(max_steps, desc="Rollout", leave=False):
        # Process one step in the environment
        # one step consists of selecting an action, taking a step in the environment,
        # and updating the context with the new observation, action, reward, and value.
        if t % plan_every == 0:
            # every plan_every steps, we generate a new plan using beam search
            # and store the predicted tokens in plan_buffer.
            # higher plan_every means we are using the same plan for longer
            # as a result, we are putting more trust in the model's prediction
            # of the future states, actions, and rewards.

            context_idx = (
                ((t + 1) * transition_dim) - action_dim - reward_dim - value_dim
            )
            context_not_dones = context[~dones, :context_idx]

            # generate a new plan using beam search
            # predicted_tokens shape (num_envs, beam_steps * transition_dim)
            predicted_tokens = vec_beam_plan(
                model,
                discretizer,
                context_not_dones,
                beam_width,
                beam_steps,
                beam_context,
                sample_expansion,
                observation_dim,
                action_dim,
                reward_dim,
                value_dim,
                transition_dim,
                obs_top_k=obs_top_k,
                act_top_k=act_top_k,
                rew_top_k=rew_top_k,
                temperature=temperature,
                greedy=greedy,
            )
            plan_buffer = torch.zeros(
                env.num_envs,
                predicted_tokens.shape[-1],
                device=device,
                dtype=predicted_tokens.dtype,
            )
            plan_buffer[~dones] = predicted_tokens
        else:
            # if we are not generating a new plan, we use the plan_buffer
            # to get the next transition_dim number of tokens
            plan_buffer = plan_buffer[:, transition_dim:]

        # get the action from the predicted tokens
        # action_token shape (num_envs, action_dim)
        action_token = plan_buffer[:, :action_dim].cpu().numpy()
        # decode the action
        # action shape (num_envs, action_dim)
        action = discretizer.decode(
            action_token, subslice=(observation_dim, observation_dim + action_dim)
        )
        action = unflatten_space(action, env.action_space)
        next_obs, reward, _, info = env.step(action)
        done = [i["success"] for i in info]
        # next_obs shape (num_envs, observation_dim)
        next_obs = flatten_space(next_obs, env.observation_space)
        # discretize the next observation
        # next_obs_token shape (num_envs, observation_dim)
        next_obs_token = discretizer.encode(
            next_obs[~dones], subslice=(0, observation_dim)
        )
        # discretize the reward and value
        # reward_value_tokens shape (num_envs, reward_dim + value_dim)
        reward_value_tokens = discretizer.encode(
            np.hstack([reward.reshape(-1, reward_dim), value_placeholder]),
            subslice=(transition_dim - reward_dim - action_dim, transition_dim),
        )

        # update the context
        context_idx = t * transition_dim
        # add action
        context[
            ~dones,
            context_idx + observation_dim : context_idx + observation_dim + action_dim,
        ] = torch.as_tensor(action_token[~dones], device=device)
        # add reward and value
        context[
            ~dones,
            context_idx + observation_dim + action_dim : context_idx + transition_dim,
        ] = torch.as_tensor(reward_value_tokens[~dones], device=device)
        # add next observation
        context[
            ~dones,
            context_idx
            + transition_dim : context_idx
            + transition_dim
            + observation_dim,
        ] = torch.as_tensor(next_obs_token, device=device)

        total_rewards[~dones] += reward[~dones]

        dones[done] = True
        if np.all(dones):
            break
    return total_rewards, dones

# %%
def calculate_loss(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    vocab_size: int,
    transition_dim: int,
    observation_dim: int,
    action_dim: int,
    reward_dim: int,
    value_dim: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    # inputs shape (batch_size, seq_len)
    # targets shape (batch_size, seq_len)
    # loss_pad_mask shape (batch_size, seq_len)
    inputs, targets, loss_pad_mask = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    loss_pad_mask = loss_pad_mask.to(device)
    # logits shape (batch_size, seq_len, vocab_size)
    logits, _ = model(inputs)
    # flatten the logits and targets to shape (batch_size * seq_len, vocab_size)
    assert logits.shape[-1] == vocab_size, "vocab_size mismatch"
    logits = logits.reshape(-1, vocab_size)
    # flatten the targets to shape (batch_size * seq_len)
    targets = targets.reshape(-1)
    # loss shape (batch_size * seq_len)
    loss = F.cross_entropy(logits, targets, reduction="none")
    assert loss.shape == (inputs.shape[0] * inputs.shape[1],), "loss shape mismatch"

    n_states = math.ceil(inputs.shape[1] / transition_dim)
    # weights shape (observation_dim + action_dim + reward_dim + value_dim)
    weights = torch.cat(
        [
            torch.ones(observation_dim, device=inputs.device),
            torch.ones(action_dim, device=inputs.device) * 1,
            torch.ones(reward_dim, device=inputs.device) * 1,
            torch.ones(value_dim, device=inputs.device) * 1,
        ]
    )
    weights = weights.repeat(n_states)[1:].repeat(inputs.shape[0], 1)
    loss = loss * weights.view(-1)
    # apply the loss pad mask to the loss because we don't want to calculate the loss for padded values
    loss = (loss * loss_pad_mask.view(-1)).mean()
    return loss


def vec_eval(
    env: Env,
    model: nn.Module,
    discretizer: KBinsDiscretizer,
    num_episodes: int,
    beam_width: int,
    beam_steps: int,
    beam_context: int,
    sample_expansion: int,
    observation_dim: int,
    action_dim: int,
    reward_dim: int,
    value_dim: int,
    transition_dim: int,
    plan_every: int,
    obs_top_k: Optional[int] = None,
    act_top_k: Optional[int] = None,
    rew_top_k: Optional[int] = None,
    temperature: float = 1.0,
    greedy: bool = False,
    device: torch.device = torch.device("cpu"),
):
    model.eval()

    # create a vectorized environment, this allows us to run multiple environments in parallel
    vec_env = DummyVecEnv([lambda: gym.make(env.name) for _ in range(num_episodes)])
    start_time = time.time()
    total_rewards, dones = vec_rollout(
        model,
        vec_env,
        discretizer,
        beam_width,
        beam_steps,
        beam_context,
        sample_expansion,
        observation_dim,
        action_dim,
        reward_dim,
        value_dim,
        transition_dim,
        (
            vec_env.envs[0]._max_episode_steps
            if not local
            else vec_env.envs[0]._max_episode_steps
        ),
        plan_every,
        obs_top_k=obs_top_k,
        act_top_k=act_top_k,
        rew_top_k=rew_top_k,
        temperature=temperature,
        greedy=greedy,
        device=device,
    )
    end_time = time.time()
    mean_rewards = np.mean(total_rewards)
    std_rewards = np.std(total_rewards)

    done_ratio = np.mean(dones)

    model.train()
    return mean_rewards, std_rewards, done_ratio, end_time - start_time, 0


def calculate_predictive_accuracy(
    model: nn.Module,
    dataloader: Subset,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0

    # sample 10% of the data
    sampling_rate = 0.1
    dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=SubsetRandomSampler(
            np.random.choice(
                len(dataloader.dataset),
                int(sampling_rate * len(dataloader.dataset)),
                replace=False,
            )
        ),
    )

    for data in tqdm(dataloader, desc="Calculating predictive accuracy", leave=False):
        x, y, mask = data
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        # (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        mask = mask.reshape(-1)
        # only consider the tokens that are not masked
        mask_idx = mask.nonzero(as_tuple=True)[0]
        y = y[mask_idx]
        logits = logits[mask_idx]
        # (batch_size * seq_len) -> (batch_size)
        y_pred = torch.argmax(logits, dim=-1)
        correct = (y_pred == y).sum().item()
        total_correct += correct
        total_samples += y.shape[0]
    model.train()
    return total_correct / total_samples


def get_optimizer(model, weight_decay, learning_rate, betas):
    param_groups = weight_decay_groups(
        model=model,
        whitelist_modules=(torch.nn.Linear, torch.nn.MultiheadAttention, EinLinear),
        blacklist_modules=(torch.nn.LayerNorm, torch.nn.Embedding),
        blacklist_named=("positional_embedding",),
    )
    optim_groups = [
        {"params": param_groups["decay"], "weight_decay": weight_decay},
        {"params": param_groups["nodecay"], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    return optimizer


def get_scheduler(optimizer, warmup_tokens, final_tokens):
    scheduler = GPTScheduler(
        optimizer,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        decay=True,
    )
    return scheduler


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    discretizer: KBinsDiscretizer,
    optimizer: torch.optim.Optimizer,
    scheduler: GPTScheduler,
    vocab_size: int,
    n_epochs: int,
    writer: SummaryWriter,
    device: torch.device = torch.device("cpu"),
    eval_every: int = 10,
    checkpoint_path: Optional[str] = None,
    clip_grad: Optional[float] = None,
):
    model.train()
    step = 0
    for epoch in trange(n_epochs, desc="Training"):
        start_time = time.time()
        total_loss = 0
        for batch in tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False
        ):
            loss = calculate_loss(
                model,
                batch,
                vocab_size,
                device=device,
                transition_dim=transition_dim,
                observation_dim=observation_dim,
                action_dim=action_dim,
                reward_dim=reward_dim,
                value_dim=value_dim,
            )

            _batch_tokens = batch[0].reshape(-1).shape[0]

            # write learning rate to tensorboard
            writer.add_scalar(
                "Learning rate", scheduler.get_current_lr(), step
            )

            scheduler.step(batch_size=_batch_tokens)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), step)
            total_loss += loss.item()
            step += 1
        writer.add_scalar("Epoch", epoch, epoch)
        end_time = time.time()
        writer.add_scalar("Time/train", end_time - start_time, epoch)

        train_accuracy = calculate_predictive_accuracy(
            model, train_dataloader, device=device
        )
        test_accuracy = calculate_predictive_accuracy(
            model, test_dataloader, device=device
        )
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), epoch)
                writer.add_histogram(
                    f"gradients/{name}", param.grad.cpu().numpy(), epoch
                )

        if epoch % eval_every == 0:
            start_time = time.time()

            (
                mean_rewards,
                std_rewards,
                done_ratio,
                mean_rollout_time,
                std_rollout_time,
            ) = vec_eval(
                env,
                model,
                discretizer,
                num_episodes=5 if not local else 1,
                beam_width=128 if not local else 2,
                beam_steps=15 if not local else 2,
                beam_context=5 if not local else 2,
                sample_expansion=2 if not local else 1,
                observation_dim=observation_dim,
                action_dim=action_dim,
                reward_dim=reward_dim,
                value_dim=value_dim,
                transition_dim=transition_dim,
                plan_every=1 if not local else 2,
                obs_top_k=1,
                act_top_k=1,
                rew_top_k=None,
                temperature=1.0,
                greedy=False,
                device=device,
            )
            writer.add_scalar("Reward/mean", mean_rewards, epoch)
            writer.add_scalar("Reward/std", std_rewards, epoch)
            writer.add_scalar("Done ratio", done_ratio, epoch)
            writer.add_scalar("Rollout time/mean", mean_rollout_time, epoch)
            writer.add_scalar("Rollout time/std", std_rollout_time, epoch)

            end_time = time.time()
            writer.add_scalar("Time/eval", end_time - start_time, epoch)

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path + "model.pth")
            torch.save(optimizer.state_dict(), checkpoint_path + "optimizer.pth")

        writer.flush()

# %%
print(f"Using device: {device}")
writer = SummaryWriter()


dataset.discretizer.to(device)
# split the dataset into train and test
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=8 if not local else 0,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)
model = TrajectoryTransformer(
    seq_len,
    embedding_dim,
    n_heads,
    transition_dim,
    n_blocks,
    vocab_size,
    use_sep_heads=use_sep_heads,
).to(device)

warmup_tokens = len(train_dataset) * seq_len
final_tokens = n_epochs * warmup_tokens

# write hyper parameters
writer.add_hparams(
    {
        "seq_len": seq_len,
        "embedding_dim": embedding_dim,
        "n_heads": n_heads,
        "transition_dim": transition_dim,
        "n_blocks": n_blocks,
        "vocab_size": vocab_size,
        "use_sep_heads": use_sep_heads,
        "weight_decay": weight_decay,
        "lr": lr,
        "betas": torch.tensor(betas),
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "eval_every": eval_every,
        "clip_grad": clip_grad,
        "warmup_tokens": warmup_tokens,
        "final_tokens": final_tokens,
        "env_name": env.name,
    },
    {},
)

optimizer = get_optimizer(model, weight_decay, lr, betas)
scheduler = get_scheduler(optimizer, warmup_tokens, final_tokens)
if load_checkpoint and os.path.exists(checkpoint_path + "model.pth"):
    print(f"Loading model from {checkpoint_path}")
    model.load_state_dict(
        torch.load(checkpoint_path + "model.pth", map_location=device)
    )
    optimizer.load_state_dict(
        torch.load(checkpoint_path + "optimizer.pth", map_location=device)
    )
else:
    train(
        model,
        train_dataloader,
        test_dataloader,
        dataset.discretizer,
        optimizer,
        scheduler,
        vocab_size,
        n_epochs,
        writer,
        device=device,
        eval_every=eval_every,
        checkpoint_path=checkpoint_path,
        clip_grad=clip_grad,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

# %%
(
    mean_rewards,
    std_rewards,
    done_ratio,
    mean_rollout_time,
    std_rollout_time,
) = vec_eval(
    env,
    model,
    dataset.discretizer,
    num_episodes=20 if not local else 10,
    beam_width=128 if not local else 2,
    beam_steps=15 if not local else 5,
    beam_context=2 if not local else 2,
    sample_expansion=2 if not local else 1,
    observation_dim=observation_dim,
    action_dim=action_dim,
    reward_dim=reward_dim,
    value_dim=value_dim,
    transition_dim=transition_dim,
    plan_every=2 if not local else 1,
    obs_top_k=1,
    act_top_k=1,
    rew_top_k=None,
    temperature=1.0,
    greedy=False,
    device=device,
)

# %%
# print are in diff cell because of the tqdm progress bar they get truncated
print(f"Mean rewards: {mean_rewards}")
print(f"Std rewards: {std_rewards}")
print(f"Done ratio: {done_ratio}")
print(f"Mean rollout time: {mean_rollout_time}")
print(f"Std rollout time: {std_rollout_time}")


