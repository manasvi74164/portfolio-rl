#!/usr/bin/env python3
"""
RL Portfolio Optimization v2 — LSTM Backbone + Bug Fixes
Optimized for: RTX 4070 GPU | Intel i9 CPU | High-RAM system

GPU OPTIMIZATIONS:
  - TF32 tensor cores enabled (RTX 4070 has 3rd-gen tensor cores)
  - torch.compile() on all networks (~20-40% faster forward/backward)
  - AMP (float16 autocast) on all inference and update steps
  - pin_memory=True + non_blocking=True for all CPU->GPU transfers
  - Larger MINI_BATCH (256) and BATCH_SIZE (2048) for better GPU occupancy

CPU OPTIMIZATIONS:
  - N_ENVS=16 — more parallel rollout environments (uses more i9 cores)
  - ThreadPoolExecutor for genuinely parallel env stepping (numpy releases GIL)
  - Threaded VecEnv replaces the original DummyVecEnv

RAM OPTIMIZATIONS:
  - Pre-allocated pinned-memory tensors for rollout buffer
  - TD3 replay buffer uses contiguous float32 arrays (cache-friendly)
  - All features precomputed once in FeatureEngine (no recomputation)
  - set_to_none=True on zero_grad() avoids allocating zero tensors
"""

import warnings
warnings.filterwarnings("ignore")

import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import torch.cuda.amp as amp

import yfinance as yf
from collections import deque
from typing import List, Tuple, Dict, Optional
import copy
from concurrent.futures import ThreadPoolExecutor

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# HARDWARE SETUP — RTX 4070 + i9
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    # cudnn autotuner: finds fastest conv/LSTM kernel for fixed input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True

    # TF32: uses tensor cores for float32 matmul with negligible precision loss
    # RTX 4070 has 3rd-gen tensor cores — this is a free ~2x matmul speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # Tell PyTorch to use TF32 (tensor core) precision for matmul
    torch.set_float32_matmul_precision("high")

    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

CPU_CORES = os.cpu_count() or 8
print(f"CPU  : {CPU_CORES} logical cores")
print(f"Using: {DEVICE}")

if not torch.cuda.is_available():
    print("\n" + "!"*60)
    print("WARNING: No CUDA GPU detected — running on CPU (slow).")
    print("To enable your RTX 4070, install CUDA PyTorch:")
    print("  pip uninstall torch torchvision torchaudio -y")
    print("  pip install torch torchvision torchaudio "
          "--index-url https://download.pytorch.org/whl/cu124")
    print("!"*60 + "\n")

# =============================================================================
# GLOBAL CONSTANTS (tuned for RTX 4070 + i9)
# =============================================================================
ASSETS       = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
N            = len(ASSETS)
N_ACTIONS    = N + 1                       # 6 (assets + cash)
P0           = 100_000.0
SEQ_LEN      = 30
N_INDICATORS = 2
N_LOOKBACKS  = 4
STATIC_DIM   = N + N + 1 + 1 + (N * N_LOOKBACKS * N_INDICATORS)   # 52
SEQ_FEAT_DIM = N * 2                       # 10
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2

TRAIN_START  = "2018-01-01"
TRAIN_END    = "2022-12-31"
VAL_START    = "2023-01-01"
VAL_END      = "2023-12-31"
TEST_START   = "2024-01-01"
TEST_END     = "2024-12-31"

# Reward / constraint params
BETA         = 0.005
XI           = 0.0025
ALPHA_CVAR   = 0.95
ZETA         = 0.005        # tightened from 0.02 (bug fix 1)
GAMMA        = 1.0
RHO_0        = 0.001
BETA_RHO     = 1.008
RHO_MAX      = 10.0         # cap rho (bug fix 3)
LAMBDA_0     = 0.0
ETA_NU       = 1e-3
ETA_LAMBDA   = 1e-3

TOTAL_STEPS  = 500_000
EPISODE_LEN  = 252          # fixed to 1 trading year
LR           = 1e-4

# GPU occupancy: RTX 4070 tensor cores want batch >= 128
# Increased from original 1024->2048 (BATCH) and 64->256 (MINI_BATCH)
BATCH_SIZE   = 2_048
MINI_BATCH   = 256

# More envs = more CPU cores utilized + bigger vectorized GPU batches
N_ENVS       = min(16, max(4, CPU_CORES // 2))

PPO_N_STEPS  = 2_048
PPO_EPOCHS   = 10
PPO_CLIP     = 0.2
PPO_ENT_COEF = 0.01

TD3_BUFFER   = 1_000_000
TD3_WARMUP   = 10_000
TD3_POLICY_FREQ = 2
TD3_TAU      = 0.005
KAPPA        = 1.5

print(f"\nConfig: N_ENVS={N_ENVS} | MINI_BATCH={MINI_BATCH} | BATCH_SIZE={BATCH_SIZE}")
print(f"STATIC_DIM={STATIC_DIM}, SEQ_FEAT_DIM={SEQ_FEAT_DIM}, SEQ_LEN={SEQ_LEN}")


# =============================================================================
# DATA DOWNLOAD
# =============================================================================
def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    print(f"\nDownloading {tickers} from {start} to {end}...")
    raw    = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    print(f"Downloaded {len(prices)} trading days, shape: {prices.shape}")
    return prices


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)

def rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu    = df.rolling(window).mean()
    sigma = df.rolling(window).std()
    return ((df - mu) / (sigma + 1e-8)).fillna(0.0)

def compute_momentum(prices: pd.DataFrame, L: int) -> pd.DataFrame:
    return ((prices - prices.shift(L)) / (prices.shift(L) + 1e-8)).fillna(0.0)

def compute_volatility(returns: pd.DataFrame, L: int) -> pd.DataFrame:
    return (returns.rolling(L).std() * np.sqrt(252)).fillna(0.0)


class FeatureEngine:
    """
    Precomputes ALL features at startup as contiguous float32 numpy arrays.
    No pandas overhead in the hot training loop — only fast numpy slicing.
    """

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = compute_returns(prices)
        self.prices_zscored = rolling_zscore(prices, 30)

        try:
            vol_raw = yf.download(ASSETS, start="2018-01-01", end="2024-12-31",
                                  auto_adjust=True, progress=False)["Volume"]
            self.volume_zscored = rolling_zscore(
                vol_raw.reindex(prices.index).fillna(0), 30)
        except Exception:
            self.volume_zscored = pd.DataFrame(
                np.zeros((len(prices), N)), index=prices.index, columns=ASSETS)

        self.lookbacks  = [21, 63, 126, 252]
        self.momentum   = {L: compute_momentum(prices, L)         for L in self.lookbacks}
        self.volatility = {L: compute_volatility(self.returns, L) for L in self.lookbacks}

        # Cache everything as float32 numpy — avoids pandas overhead in hot loop
        self._returns_np = self.returns.values.astype(np.float32)
        self._volume_np  = self.volume_zscored.values.astype(np.float32)
        self._mom_np     = {L: self.momentum[L].values.astype(np.float32)
                            for L in self.lookbacks}
        self._vol_np     = {L: self.volatility[L].values.astype(np.float32)
                            for L in self.lookbacks}

        self.dates  = prices.index
        self.n_days = len(prices)
        print(f"FeatureEngine ready: {self.n_days} days")

    def get_state(self, t: int, prev_weights: np.ndarray,
                  current_weights: np.ndarray, cash_balance: float,
                  constraint_state: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          seq_features:    (SEQ_LEN, SEQ_FEAT_DIM) = (30, 10)
          static_features: (STATIC_DIM,)            = (52,)
        """
        t_start    = max(0, t - SEQ_LEN + 1)
        ret_window = self._returns_np[t_start:t+1]
        vol_window = self._volume_np[t_start:t+1]

        if ret_window.shape[0] < SEQ_LEN:
            pad        = SEQ_LEN - ret_window.shape[0]
            ret_window = np.vstack([np.zeros((pad, N), dtype=np.float32), ret_window])
            vol_window = np.vstack([np.zeros((pad, N), dtype=np.float32), vol_window])

        seq_features = np.clip(
            np.concatenate([ret_window, vol_window], axis=1), -5.0, 5.0)

        prev_w = prev_weights.astype(np.float32)
        cur_w  = current_weights.astype(np.float32)
        cash_n = np.array([cash_balance / P0], dtype=np.float32)
        cs     = np.array([np.clip(constraint_state, -10, 10)], dtype=np.float32)

        ind_list = []
        for L in self.lookbacks:
            ind_list.extend([
                np.clip(self._mom_np[L][t], -2,  2),
                np.clip(self._vol_np[L][t],  0,  5),
            ])
        static_features = np.concatenate([prev_w, cur_w, cash_n, cs,
                                           np.concatenate(ind_list)])
        return seq_features, static_features


# =============================================================================
# ENVIRONMENT
# =============================================================================
class PortfolioEnv:
    def __init__(self, feature_engine: FeatureEngine,
                 date_indices: List[int],
                 episode_len: int = EPISODE_LEN,
                 mode: str = "train"):
        self.fe           = feature_engine
        self.date_indices = date_indices
        self.episode_len  = episode_len
        self.mode         = mode
        self._reset_internal()

    def _reset_internal(self):
        self.portfolio_value  = P0
        self.cash_balance     = P0
        self.weights          = np.zeros(N_ACTIONS, dtype=np.float32)
        self.weights[-1]      = 1.0
        self.prev_weights     = self.weights.copy()
        self.constraint_state = 0.0
        self.portfolio_returns_history = []
        self.nu          = 0.0
        self.lambda_cvar = 0.0
        self.rho_alm     = RHO_0

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._reset_internal()
        max_start = len(self.date_indices) - self.episode_len - 1
        start_pos = random.randint(0, max(0, max_start))
        self.t_idx  = self.date_indices[start_pos]
        self.t_step = 0
        return self._get_state()

    def _get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fe.get_state(
            t=self.t_idx,
            prev_weights=self.prev_weights[:N],
            current_weights=self.weights[:N],
            cash_balance=self.cash_balance,
            constraint_state=self.constraint_state,
        )

    def step(self, action: np.ndarray) -> Tuple[Tuple, float, bool, dict]:
        action = np.clip(action, 1e-8, 1.0)
        action = action / action.sum()

        t_next = self.t_idx + 1
        if t_next >= len(self.fe.prices):
            return self._get_state(), 0.0, True, {}

        r_assets    = np.clip(self.fe._returns_np[t_next], -0.5, 0.5)
        w_assets    = action[:N]
        r_portfolio = float(np.dot(w_assets, r_assets))

        self.portfolio_returns_history.append(r_portfolio)
        hist     = self.portfolio_returns_history[-30:]
        sigma_sq = float(np.var(hist)) if len(hist) > 1 else 0.0

        turnover         = float(np.sum(np.abs(action - self.weights)))
        transaction_cost = XI * turnover

        excess       = max(0.0, -r_portfolio - self.nu)
        C_estimate   = excess / (1.0 - ALPHA_CVAR)
        cvar_penalty = (self.lambda_cvar * C_estimate
                        + (self.rho_alm / 2.0) * (C_estimate ** 2))

        reward = float(np.clip(
            r_portfolio - BETA * sigma_sq - transaction_cost - cvar_penalty,
            -1.0, 1.0))   # bug fix 2: clamp reward

        self.constraint_state = 0.99 * self.constraint_state + C_estimate

        self.prev_weights     = self.weights.copy()
        self.weights          = action.copy()
        self.portfolio_value *= (1.0 + r_portfolio)
        self.cash_balance     = self.portfolio_value * action[-1]

        self.t_idx  += 1
        self.t_step += 1
        done = (self.t_step >= self.episode_len)

        info = {
            "r_portfolio":     r_portfolio,
            "sigma_sq":        sigma_sq,
            "turnover":        turnover,
            "C_estimate":      C_estimate,
            "portfolio_value": self.portfolio_value,
        }
        return self._get_state(), reward, done, info

    def set_alm_params(self, nu: float, lambda_cvar: float, rho_alm: float):
        self.nu, self.lambda_cvar, self.rho_alm = nu, lambda_cvar, rho_alm


# =============================================================================
# THREADED VECTORIZED ENV
# Runs N_ENVS environments in parallel using a thread pool.
# numpy operations release the GIL -> real parallel speedup on i9 cores.
# =============================================================================
class ThreadedVecEnv:
    def __init__(self, env_fns):
        self.envs     = [fn() for fn in env_fns]
        self.n        = len(self.envs)
        self.executor = ThreadPoolExecutor(max_workers=self.n)

    def reset(self):
        futures = [self.executor.submit(e.reset) for e in self.envs]
        states  = [f.result() for f in futures]
        return (np.stack([s[0] for s in states]),
                np.stack([s[1] for s in states]))

    def step(self, actions: np.ndarray):
        futures = [self.executor.submit(e.step, a)
                   for e, a in zip(self.envs, actions)]
        results = [f.result() for f in futures]
        next_states, rewards, dones, infos = zip(*results)

        seqs    = np.stack([s[0] for s in next_states])
        statics = np.stack([s[1] for s in next_states])
        rewards = np.array(rewards, dtype=np.float32)
        dones   = np.array(dones,   dtype=bool)

        # Auto-reset done envs in parallel
        reset_futs = {i: self.executor.submit(self.envs[i].reset)
                      for i, d in enumerate(dones) if d}
        for i, fut in reset_futs.items():
            s           = fut.result()
            seqs[i]    = s[0]
            statics[i] = s[1]

        return (seqs, statics), rewards, dones, list(infos)

    def set_alm_params(self, nu, lam, rho):
        for e in self.envs:
            e.set_alm_params(nu, lam, rho)

    def __del__(self):
        self.executor.shutdown(wait=False)


def make_env(date_indices, feature_engine, mode="train"):
    def _init():
        return PortfolioEnv(feature_engine, date_indices,
                            episode_len=EPISODE_LEN, mode=mode)
    return _init


# =============================================================================
# MANAGERS
# =============================================================================
class CVaRManager:
    def __init__(self, alpha=ALPHA_CVAR, zeta=ZETA,
                 buffer_size=100, device=DEVICE):
        self.alpha    = alpha
        self.zeta     = zeta
        self.device   = device
        self.nu       = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.lam      = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.z_buffer = deque(maxlen=buffer_size)

    def trajectory_cost(self, episode_returns: List[float]) -> float:
        n   = len(episode_returns)
        cum = float(np.prod([1 + r for r in episode_returns]) - 1)
        Z   = -((1 + cum) ** (252 / max(n, 1)) - 1) if n > 0 else 0.0
        self.z_buffer.append(Z)
        return Z

    def compute_H(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z      = (torch.tensor([0.0], device=self.device) if not self.z_buffer
                  else torch.tensor(list(self.z_buffer), device=self.device,
                                    dtype=torch.float32))
        excess = torch.relu(z - self.nu)
        H      = self.nu + excess.mean() / (1.0 - self.alpha)
        return H, H - self.zeta

    def update(self):
        if len(self.z_buffer) < 2:
            return
        Z        = torch.tensor(list(self.z_buffer), device=self.device,
                                dtype=torch.float32)
        p_exceed = (Z >= self.nu).float().mean()
        self.nu  = self.nu - ETA_NU * (1.0 - p_exceed / (1.0 - self.alpha))
        _, C     = self.compute_H()
        self.lam = torch.clamp(self.lam + ETA_LAMBDA * C, min=0.0)

    @property
    def nu_val(self):  return float(self.nu.item())
    @property
    def lam_val(self): return float(self.lam.item())


class ALMManager:
    def __init__(self, rho0=RHO_0, beta_rho=BETA_RHO,
                 lambda0=LAMBDA_0, device=DEVICE):
        self.rho = rho0; self.beta_rho = beta_rho; self.lam = lambda0

    def alm_loss(self, C: torch.Tensor) -> torch.Tensor:
        return self.lam * C + (self.rho / 2.0) * C ** 2

    def update(self, C: float):
        self.lam = max(0.0, self.lam + self.rho * C)
        if C > 0:                                   # bug fix 3b
            self.rho = min(self.beta_rho * self.rho, RHO_MAX)

    @property
    def lam_val(self): return self.lam
    @property
    def rho_val(self): return self.rho


# =============================================================================
# NETWORKS
# =============================================================================
class LSTMActorNetwork(nn.Module):
    def __init__(self, seq_feat_dim=SEQ_FEAT_DIM, static_dim=STATIC_DIM,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
                 action_dim=N_ACTIONS):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=seq_feat_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0)
        in_dim = lstm_hidden + static_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512),    nn.ReLU(),          nn.Dropout(0.1),
            nn.Linear(512, action_dim))

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        seq    = torch.nan_to_num(seq,    nan=0.0, posinf=5.0, neginf=-5.0)
        static = torch.nan_to_num(static, nan=0.0, posinf=5.0, neginf=-5.0)
        _, (h_n, _) = self.lstm(seq)
        combined    = torch.cat([h_n[-1], static], dim=-1)
        logits      = torch.clamp(self.head(combined), -10.0, 10.0)
        alpha       = torch.nan_to_num(F.softplus(logits) + 1e-6, nan=1.0)
        return torch.clamp(alpha, min=1e-3, max=1e3)

    def get_action(self, seq, static, explore=True, kappa=KAPPA):
        alpha = self.forward(seq, static)
        dist  = Dirichlet(torch.clamp(kappa * alpha if explore else alpha,
                                      min=1e-3, max=1e3))
        action   = dist.rsample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def get_deterministic_action(self, seq, static):
        alpha = self.forward(seq, static)
        return alpha / alpha.sum(dim=-1, keepdim=True)


class LSTMCriticNetwork(nn.Module):
    def __init__(self, seq_feat_dim=SEQ_FEAT_DIM, static_dim=STATIC_DIM,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
                 action_dim=N_ACTIONS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_feat_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0)
        in_dim = lstm_hidden + static_dim + action_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512),    nn.ReLU(),          nn.Dropout(0.1),
            nn.Linear(512, 1))

    def forward(self, seq, static, action):
        seq    = torch.nan_to_num(seq,    nan=0.0, posinf=5.0, neginf=-5.0)
        static = torch.nan_to_num(static, nan=0.0, posinf=5.0, neginf=-5.0)
        _, (h_n, _) = self.lstm(seq)
        combined    = torch.cat([h_n[-1], static, action], dim=-1)
        return self.head(combined)


class LSTMValueNetwork(nn.Module):
    def __init__(self, seq_feat_dim=SEQ_FEAT_DIM, static_dim=STATIC_DIM,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_feat_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0)
        in_dim = lstm_hidden + static_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512),    nn.ReLU(),          nn.Dropout(0.1),
            nn.Linear(512, 1))

    def forward(self, seq, static):
        seq    = torch.nan_to_num(seq,    nan=0.0, posinf=5.0, neginf=-5.0)
        static = torch.nan_to_num(static, nan=0.0, posinf=5.0, neginf=-5.0)
        _, (h_n, _) = self.lstm(seq)
        combined    = torch.cat([h_n[-1], static], dim=-1)
        return self.head(combined)


def maybe_compile(model: nn.Module) -> nn.Module:
    """
    torch.compile() requires Triton which has NO Windows support.
    Skipped automatically on Windows. All other GPU optimizations
    (AMP, TF32 tensor cores, pinned memory) remain fully active.
    On Linux/Mac this gives an additional ~20-40% speedup.
    """
    import sys
    if sys.platform == "win32":
        print(f"  torch.compile() skipped (Triton not supported on Windows)")
        return model
    major = int(torch.__version__.split(".")[0])
    if major < 2:
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print(f"  torch.compile() OK: {type(model).__name__}")
        return compiled
    except Exception as e:
        print(f"  torch.compile() skipped ({type(e).__name__})")
    return model


# =============================================================================
# ROLLOUT BUFFER — pinned memory for fast CPU->GPU DMA
# =============================================================================
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, seq_len, seq_feat_dim,
                 static_dim, action_dim, device=DEVICE):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.device  = device
        pin = {"pin_memory": True} if device.type == "cuda" else {}
        # Pre-allocated pinned (page-locked) host memory -> fast DMA to GPU
        self.seqs      = torch.zeros(n_steps, n_envs, seq_len, seq_feat_dim,  **pin)
        self.statics   = torch.zeros(n_steps, n_envs, static_dim,             **pin)
        self.actions   = torch.zeros(n_steps, n_envs, action_dim,             **pin)
        self.rewards   = torch.zeros(n_steps, n_envs,                         **pin)
        self.log_probs = torch.zeros(n_steps, n_envs,                         **pin)
        self.values    = torch.zeros(n_steps, n_envs,                         **pin)
        self.dones     = torch.zeros(n_steps, n_envs,                         **pin)
        self.ptr = 0

    def add(self, seq, static, action, reward, log_prob, value, done):
        i = self.ptr
        self.seqs[i].copy_(torch.from_numpy(seq))
        self.statics[i].copy_(torch.from_numpy(static))
        self.actions[i].copy_(torch.from_numpy(action))
        self.rewards[i]   = torch.from_numpy(reward)
        self.log_probs[i] = torch.from_numpy(log_prob)
        self.values[i]    = torch.from_numpy(value)
        self.dones[i]     = torch.from_numpy(done)
        self.ptr = (self.ptr + 1) % self.n_steps

    def get_batches(self, batch_size, last_values, gamma=GAMMA):
        T, E       = self.n_steps, self.n_envs
        gae_lambda = 0.95
        values_np  = self.values.numpy()
        rewards_np = self.rewards.numpy()
        dones_np   = self.dones.numpy()

        advantages = np.zeros((T, E), dtype=np.float32)
        last_gae   = np.zeros(E, dtype=np.float32)
        for t in reversed(range(T)):
            nv       = last_values if t == T - 1 else values_np[t + 1]
            delta    = (rewards_np[t] + gamma * nv * (1 - dones_np[t])
                        - values_np[t])
            last_gae = (delta + gamma * gae_lambda
                        * (1 - dones_np[t]) * last_gae)
            advantages[t] = last_gae

        returns = advantages + values_np

        # Transfer entire arrays to GPU in one shot (non-blocking DMA)
        flat_seqs  = self.seqs.view(-1, SEQ_LEN, SEQ_FEAT_DIM).to(
            self.device, non_blocking=True)
        flat_stats = self.statics.view(-1, STATIC_DIM).to(
            self.device, non_blocking=True)
        flat_acts  = self.actions.view(-1, N_ACTIONS).to(
            self.device, non_blocking=True)
        flat_rets  = torch.from_numpy(returns.reshape(-1)).to(
            self.device, non_blocking=True)
        flat_advs  = torch.from_numpy(advantages.reshape(-1)).to(
            self.device, non_blocking=True)
        flat_lps   = self.log_probs.view(-1).to(
            self.device, non_blocking=True)

        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)

        n   = flat_seqs.shape[0]
        idx = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            yield (flat_seqs[b], flat_stats[b], flat_acts[b],
                   flat_rets[b], flat_advs[b], flat_lps[b])


# =============================================================================
# HELPERS
# =============================================================================
def annualized_return(daily_returns: List[float]) -> float:
    n = len(daily_returns)
    if n == 0:
        return 0.0
    cum = float(np.prod([1 + r for r in daily_returns]) - 1)
    return ((1 + cum) ** (252 / n) - 1) * 100


# =============================================================================
# PPO TRAINING
# =============================================================================
def train_ppo(experiment_name: str,
              feature_engine: FeatureEngine,
              train_idx: List[int],
              use_cvar: bool = True,
              total_steps: int = TOTAL_STEPS) -> Dict:
    print(f"\n{'='*60}")
    print(f"Training PPO | {'CVaR' if use_cvar else 'Variance'} | {experiment_name}")
    print(f"{'='*60}")

    actor  = maybe_compile(LSTMActorNetwork().to(DEVICE))
    critic = maybe_compile(LSTMValueNetwork().to(DEVICE))
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    cvar_mgr = CVaRManager()
    alm_mgr  = ALMManager()
    evar_buf = deque(maxlen=100)

    env_fns = [make_env(train_idx, feature_engine) for _ in range(N_ENVS)]
    vec_env = ThreadedVecEnv(env_fns)
    buf     = RolloutBuffer(PPO_N_STEPS, N_ENVS, SEQ_LEN, SEQ_FEAT_DIM,
                            STATIC_DIM, N_ACTIONS)

    logs = {k: [] for k in ["episode_returns", "cvar_vals", "violations",
                             "lambdas", "rhos", "sigma_sqs",
                             "turnovers", "steps"]}

    obs_seqs, obs_statics  = vec_env.reset()
    ep_rets_per_env        = [[] for _ in range(N_ENVS)]
    step, ep_count         = 0, 0
    scaler                 = amp.GradScaler()

    while step < total_steps:
        buf.ptr = 0
        for _ in range(PPO_N_STEPS):
            seq_t  = torch.from_numpy(obs_seqs).to(DEVICE, non_blocking=True)
            stat_t = torch.from_numpy(obs_statics).to(DEVICE, non_blocking=True)

            with torch.no_grad(), amp.autocast():
                alpha    = actor(seq_t, stat_t)
                dist     = Dirichlet(alpha)
                actions  = dist.rsample()
                log_prob = dist.log_prob(actions)
                values   = critic(seq_t, stat_t).squeeze(-1)

            actions_np  = actions.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            values_np   = values.cpu().numpy()

            (next_seqs, next_statics), rewards, dones, infos = \
                vec_env.step(actions_np)

            buf.add(obs_seqs, obs_statics, actions_np, rewards,
                    log_prob_np, values_np, dones.astype(np.float32))

            for i, (done, info) in enumerate(zip(dones, infos)):
                ep_rets_per_env[i].append(info.get("r_portfolio", 0.0))
                if done:
                    ep_ret = ep_rets_per_env[i].copy()
                    ep_rets_per_env[i] = []
                    cvar_mgr.trajectory_cost(ep_ret)
                    cvar_mgr.update()
                    _, C_t = cvar_mgr.compute_H()
                    C_val  = float(C_t.item())
                    alm_mgr.update(C_val)
                    vec_env.set_alm_params(cvar_mgr.nu_val,
                                           alm_mgr.lam_val,
                                           alm_mgr.rho_val)
                    evar_buf.append(C_val)

                    cum_ret = annualized_return(ep_ret)
                    logs["episode_returns"].append(cum_ret)
                    logs["cvar_vals"].append(float(cvar_mgr.compute_H()[0].item()))
                    logs["violations"].append(C_val)
                    logs["lambdas"].append(alm_mgr.lam_val)
                    logs["rhos"].append(alm_mgr.rho_val)
                    logs["sigma_sqs"].append(
                        float(np.var(ep_ret)) if len(ep_ret) > 1 else 0.0)
                    logs["turnovers"].append(info.get("turnover", 0.0))
                    logs["steps"].append(step)
                    ep_count += 1

                    if ep_count % 20 == 0:
                        avg = np.mean(logs["episode_returns"][-20:])
                        print(f"  Step {step:>7d} | Ep {ep_count:>4d} | "
                              f"AvgRet {avg:+.2f}% | "
                              f"lambda={alm_mgr.lam_val:.4f} | "
                              f"rho={alm_mgr.rho_val:.4f} | C={C_val:.4f}")

            obs_seqs, obs_statics = next_seqs, next_statics
            step += N_ENVS

        # Bootstrap value for GAE
        with torch.no_grad(), amp.autocast():
            lv = critic(
                torch.from_numpy(obs_seqs).to(DEVICE, non_blocking=True),
                torch.from_numpy(obs_statics).to(DEVICE, non_blocking=True),
            ).squeeze(-1).cpu().numpy()

        for _ in range(PPO_EPOCHS):
            for (s, st, a, ret, adv, old_lp) in buf.get_batches(MINI_BATCH, lv):
                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)

                with amp.autocast():
                    alpha_   = actor(s, st)
                    dist_    = Dirichlet(alpha_)
                    new_lp   = dist_.log_prob(a)
                    entropy  = dist_.entropy()
                    ratio    = torch.exp(new_lp - old_lp)
                    clip_r   = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                    a_loss   = -torch.min(ratio * adv, clip_r * adv).mean()
                    a_loss  -= PPO_ENT_COEF * entropy.mean()

                    if use_cvar:
                        _, C_t = cvar_mgr.compute_H()
                        a_loss += alm_mgr.alm_loss(C_t)
                    elif len(evar_buf) > 1:
                        ev     = np.mean(list(evar_buf)) + np.std(list(evar_buf))
                        a_loss += alm_mgr.lam_val * torch.tensor(ev, device=DEVICE)

                    v_loss = F.mse_loss(critic(s, st).squeeze(-1), ret)

                scaler.scale(a_loss).backward()
                scaler.unscale_(actor_opt)
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                scaler.step(actor_opt)

                scaler.scale(v_loss).backward()
                scaler.unscale_(critic_opt)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                scaler.step(critic_opt)

                scaler.update()

    print(f"PPO done. Episodes: {ep_count}")
    return {"actor": actor, "critic": critic, "logs": logs,
            "cvar_mgr": cvar_mgr, "alm_mgr": alm_mgr,
            "experiment": experiment_name}


# =============================================================================
# REPLAY BUFFER (TD3) — contiguous float32 for cache-friendly random access
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity, seq_len, seq_feat_dim, static_dim,
                 action_dim, device=DEVICE):
        self.capacity = capacity
        self.device   = device
        self.ptr = self.size = 0
        self.seqs         = np.empty((capacity, seq_len, seq_feat_dim), dtype=np.float32)
        self.statics      = np.empty((capacity, static_dim),            dtype=np.float32)
        self.actions      = np.empty((capacity, action_dim),            dtype=np.float32)
        self.rewards      = np.empty((capacity, 1),                     dtype=np.float32)
        self.next_seqs    = np.empty((capacity, seq_len, seq_feat_dim), dtype=np.float32)
        self.next_statics = np.empty((capacity, static_dim),            dtype=np.float32)
        self.dones        = np.empty((capacity, 1),                     dtype=np.float32)

    def add(self, seq, static, action, reward,
            next_seq, next_static, done):
        p = self.ptr
        self.seqs[p]         = seq
        self.statics[p]      = static
        self.actions[p]      = action
        self.rewards[p]      = reward
        self.next_seqs[p]    = next_seq
        self.next_statics[p] = next_static
        self.dones[p]        = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx  = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.from_numpy(x[idx]).to(
            self.device, non_blocking=True)
        return (to_t(self.seqs),    to_t(self.statics),
                to_t(self.actions), to_t(self.rewards),
                to_t(self.next_seqs), to_t(self.next_statics),
                to_t(self.dones))

    def __len__(self): return self.size


# =============================================================================
# TD3 TRAINING
# =============================================================================
def train_td3(experiment_name: str,
              feature_engine: FeatureEngine,
              train_idx: List[int],
              use_cvar: bool = True,
              total_steps: int = TOTAL_STEPS) -> Dict:
    print(f"\n{'='*60}")
    print(f"Training TD3 | {'CVaR' if use_cvar else 'Variance'} | {experiment_name}")
    print(f"{'='*60}")

    actor   = maybe_compile(LSTMActorNetwork().to(DEVICE))
    critic1 = maybe_compile(LSTMCriticNetwork().to(DEVICE))
    critic2 = maybe_compile(LSTMCriticNetwork().to(DEVICE))
    actor_t  = copy.deepcopy(actor)
    critic1_t = copy.deepcopy(critic1)
    critic2_t = copy.deepcopy(critic2)

    actor_opt = torch.optim.Adam(actor.parameters(),   lr=LR)
    c1_opt    = torch.optim.Adam(critic1.parameters(), lr=LR)
    c2_opt    = torch.optim.Adam(critic2.parameters(), lr=LR)

    def soft_update(tgt, src, tau=TD3_TAU):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    cvar_mgr  = CVaRManager()
    alm_mgr   = ALMManager()
    evar_buf  = deque(maxlen=100)
    replay    = ReplayBuffer(TD3_BUFFER, SEQ_LEN, SEQ_FEAT_DIM,
                             STATIC_DIM, N_ACTIONS)

    env  = PortfolioEnv(feature_engine, train_idx, episode_len=EPISODE_LEN)
    logs = {k: [] for k in ["episode_returns", "cvar_vals", "violations",
                             "lambdas", "rhos", "sigma_sqs",
                             "turnovers", "steps"]}

    obs        = env.reset()
    ep_rets    = []
    ep_turns   = []
    ep_count   = 0
    c_updates  = 0
    scaler     = amp.GradScaler()

    for step in range(total_steps):
        if step < TD3_WARMUP:
            action = np.random.dirichlet(np.ones(N_ACTIONS)).astype(np.float32)
        else:
            seq_t  = torch.from_numpy(obs[0]).unsqueeze(0).to(
                DEVICE, non_blocking=True)
            stat_t = torch.from_numpy(obs[1]).unsqueeze(0).to(
                DEVICE, non_blocking=True)
            with torch.no_grad(), amp.autocast():
                action, _, _ = actor.get_action(seq_t, stat_t, explore=True)
            action = action.squeeze(0).cpu().numpy()

        next_obs, reward, done, info = env.step(action)
        ep_rets.append(info.get("r_portfolio", 0.0))
        ep_turns.append(info.get("turnover", 0.0))
        replay.add(obs[0], obs[1], action, reward,
                   next_obs[0], next_obs[1], float(done))
        obs = next_obs

        if done:
            cvar_mgr.trajectory_cost(ep_rets)
            cvar_mgr.update()
            _, C_t = cvar_mgr.compute_H()
            C_val  = float(C_t.item())
            alm_mgr.update(C_val)
            env.set_alm_params(cvar_mgr.nu_val, alm_mgr.lam_val,
                               alm_mgr.rho_val)
            evar_buf.append(C_val)

            logs["episode_returns"].append(annualized_return(ep_rets))
            logs["cvar_vals"].append(float(cvar_mgr.compute_H()[0].item()))
            logs["violations"].append(C_val)
            logs["lambdas"].append(alm_mgr.lam_val)
            logs["rhos"].append(alm_mgr.rho_val)
            logs["sigma_sqs"].append(
                float(np.var(ep_rets)) if len(ep_rets) > 1 else 0.0)
            logs["turnovers"].append(float(np.mean(ep_turns)))
            logs["steps"].append(step)
            ep_count += 1

            if ep_count % 20 == 0:
                avg = np.mean(logs["episode_returns"][-20:])
                print(f"  Step {step:>7d} | Ep {ep_count:>4d} | "
                      f"AvgRet {avg:+.2f}% | "
                      f"lambda={alm_mgr.lam_val:.4f} | "
                      f"rho={alm_mgr.rho_val:.4f} | C={C_val:.4f}")

            ep_rets, ep_turns = [], []
            obs = env.reset()

        if len(replay) < max(TD3_WARMUP, BATCH_SIZE):
            continue

        s, st, a, r, ns, nst, d = replay.sample(BATCH_SIZE)

        # ── Critic update ─────────────────────────────────────────────────
        c1_opt.zero_grad(set_to_none=True)
        c2_opt.zero_grad(set_to_none=True)

        with amp.autocast():
            with torch.no_grad():
                na, _, _ = actor_t.get_action(ns, nst, explore=True)
                tgt = r + GAMMA * (1 - d) * torch.min(
                    critic1_t(ns, nst, na), critic2_t(ns, nst, na))
            c1l = F.mse_loss(critic1(s, st, a), tgt)
            c2l = F.mse_loss(critic2(s, st, a), tgt)

        scaler.scale(c1l).backward()
        scaler.unscale_(c1_opt)
        nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
        scaler.step(c1_opt)

        scaler.scale(c2l).backward()
        scaler.unscale_(c2_opt)
        nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
        scaler.step(c2_opt)

        scaler.update()
        c_updates += 1

        # ── Delayed actor update ──────────────────────────────────────────
        if c_updates % TD3_POLICY_FREQ == 0:
            actor_opt.zero_grad(set_to_none=True)
            with amp.autocast():
                a_curr, _, ent = actor.get_action(s, st, explore=True)
                al = (-critic1(s, st, a_curr).mean()
                      - PPO_ENT_COEF * ent.mean())
                if use_cvar:
                    _, C_t = cvar_mgr.compute_H()
                    al += alm_mgr.alm_loss(C_t)
                elif len(evar_buf) > 1:
                    ev  = np.mean(list(evar_buf)) + np.std(list(evar_buf))
                    al += alm_mgr.lam_val * torch.tensor(
                        ev, device=DEVICE, dtype=torch.float32)
            scaler.scale(al).backward()
            scaler.unscale_(actor_opt)
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            scaler.step(actor_opt)
            scaler.update()

            soft_update(actor_t,   actor)
            soft_update(critic1_t, critic1)
            soft_update(critic2_t, critic2)

    print(f"TD3 done. Episodes: {ep_count}")
    return {"actor": actor, "critic": critic1, "logs": logs,
            "cvar_mgr": cvar_mgr, "alm_mgr": alm_mgr,
            "experiment": experiment_name}


# =============================================================================
# EVALUATION
# =============================================================================
def compute_sharpe(returns, rf=0.05/252):
    r  = np.array(returns)
    ex = r - rf
    return float(np.mean(ex) / (np.std(ex) + 1e-8) * np.sqrt(252))

def compute_max_drawdown(returns):
    cum  = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-8)
    return float(np.min(dd) * 100)

def compute_cvar(returns, alpha=ALPHA_CVAR):
    r    = np.array(returns)
    var  = np.percentile(r, (1 - alpha) * 100)
    tail = r[r <= var]
    return float(np.mean(tail)) if len(tail) > 0 else float(var)


def evaluate_policy(actor, feature_engine, date_indices,
                    episode_len=None, n_episodes=3):
    actor.eval()
    episode_len = episode_len or min(EPISODE_LEN, len(date_indices) - 1)
    all_rets, all_turns = [], []

    for _ in range(n_episodes):
        env = PortfolioEnv(feature_engine, date_indices,
                           episode_len=episode_len, mode="eval")
        obs = env.reset()
        ep_rets, ep_turns = [], []

        for _ in range(episode_len):
            seq_t  = torch.from_numpy(obs[0]).unsqueeze(0).to(
                DEVICE, non_blocking=True)
            stat_t = torch.from_numpy(obs[1]).unsqueeze(0).to(
                DEVICE, non_blocking=True)
            with torch.no_grad(), amp.autocast():
                action = actor.get_deterministic_action(seq_t, stat_t)
            obs, _, done, info = env.step(action.squeeze(0).cpu().numpy())
            ep_rets.append(info.get("r_portfolio", 0.0))
            ep_turns.append(info.get("turnover", 0.0))
            if done:
                break

        all_rets.append(ep_rets)
        all_turns.append(np.mean(ep_turns))

    actor.train()
    rets    = all_rets[0]
    cum_ret = float(np.prod([1 + r for r in rets]) - 1) * 100
    return {
        "cum_return":   cum_ret,
        "sharpe":       compute_sharpe(rets),
        "max_drawdown": compute_max_drawdown(rets),
        "cvar_95":      compute_cvar(rets),
        "turnover":     float(np.mean(all_turns)),
        "returns":      rets,
    }


def compute_equal_weight_baseline(feature_engine, date_indices):
    w_eq = np.ones(N) / N
    rets = [float(np.dot(w_eq, feature_engine._returns_np[date_indices[i + 1]]))
            for i in range(len(date_indices) - 1)]
    cum_ret = float(np.prod([1 + r for r in rets]) - 1) * 100
    return {"cum_return": cum_ret, "sharpe": compute_sharpe(rets),
            "max_drawdown": compute_max_drawdown(rets),
            "cvar_95": compute_cvar(rets), "turnover": 0.0, "returns": rets}


# =============================================================================
# PLOTS & PDF
# =============================================================================
COLORS  = {"PPO_CVaR": "#1f77b4", "PPO_Var": "#ff7f0e",
           "TD3_CVaR": "#2ca02c", "TD3_Var": "#d62728",
           "EqualWeight": "#9467bd"}
LSTYLES = {"PPO_CVaR": "-", "PPO_Var": "--",
           "TD3_CVaR": "-.", "TD3_Var": ":", "EqualWeight": "-"}


def generate_plots(results, test_metrics):
    fig_list = []

    fig1, ax = plt.subplots(figsize=(12, 6))
    for name, m in test_metrics.items():
        cum = np.cumprod(1 + np.array(m["returns"])) * 100 - 100
        ax.plot(cum, label=name, color=COLORS.get(name, "gray"),
                linestyle=LSTYLES.get(name, "-"), lw=1.8)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title("Cumulative Return — Test Set (2024)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig_list.append(fig1)

    fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (name, res) in zip(axes.flatten(), results.items()):
        ep = res["logs"]["episode_returns"]
        ax.plot(ep, alpha=0.5, lw=0.8, color=COLORS.get(name, "gray"))
        w  = max(1, len(ep) // 20)
        ax.plot(pd.Series(ep).rolling(w, min_periods=1).mean(),
                color="black", lw=1.5)
        ax.set_title(f"{name} Training", fontsize=11)
        ax.set_xlabel("Episode"); ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig_list.append(fig2)

    fig3, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (name, res) in zip(axes.flatten(), results.items()):
        ax2 = ax.twinx()
        ax.plot(res["logs"]["violations"], alpha=0.7,
                color="tab:red", label="C")
        ax2.plot(res["logs"]["lambdas"], alpha=0.7,
                 color="tab:blue", ls="--", label="λ")
        ax.set_title(f"{name} Constraint", fontsize=11)
        ax.set_ylabel("C", color="tab:red")
        ax2.set_ylabel("λ", color="tab:blue")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("CVaR Violation and λ", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig_list.append(fig3)

    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    names  = list(test_metrics.keys())
    colors = [COLORS.get(n, "gray") for n in names]
    for ax, vals, title in zip(
            axes,
            [[test_metrics[n]["sharpe"] for n in names],
             [abs(test_metrics[n]["max_drawdown"]) for n in names],
             [test_metrics[n]["turnover"] for n in names]],
            ["Sharpe Ratio", "Max Drawdown (%)", "Avg Turnover"]):
        bars = ax.bar(names, vals, color=colors, edgecolor="black", lw=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.001 * max(vals),
                    f"{v:.3f}", ha="center", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Test Set Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig_list.append(fig4)

    print(f"Generated {len(fig_list)} figures.")
    return fig_list


def save_pdf(fig_list, test_metrics,
             pdf_path="rl_portfolio_report_v2.pdf"):
    with pdf_backend.PdfPages(pdf_path) as pdf:
        fig_c, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        ax.text(0.5, 0.7,  "RL Portfolio Optimization v2",
                ha="center", fontsize=20, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.55, "LSTM Backbone + Bug Fixes",
                ha="center", fontsize=14, color="#2c3e50",
                transform=ax.transAxes)
        ax.text(0.5, 0.43, "Assets: AAPL | MSFT | GOOGL | AMZN | TSLA",
                ha="center", fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.33, "Train: 2018–2022 | Val: 2023 | Test: 2024",
                ha="center", fontsize=11, transform=ax.transAxes)
        pdf.savefig(fig_c, bbox_inches="tight"); plt.close(fig_c)

        fig_t, ax = plt.subplots(figsize=(14, 4))
        ax.axis("off")
        rows = [[n,
                 f"{test_metrics[n]['cum_return']:+.2f}%",
                 f"{test_metrics[n]['sharpe']:.3f}",
                 f"{test_metrics[n]['max_drawdown']:.2f}%",
                 f"{test_metrics[n]['cvar_95']:.4f}",
                 f"{test_metrics[n]['turnover']:.4f}"]
                for n in test_metrics]
        cols = ["Experiment", "Cum.Return", "Sharpe",
                "MaxDrawdown", "CVaR_0.95", "Turnover"]
        tbl  = ax.table(cellText=rows, colLabels=cols,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor("#2c3e50")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        ax.set_title("Test Results (2024)", fontsize=13,
                     fontweight="bold", pad=20)
        pdf.savefig(fig_t, bbox_inches="tight"); plt.close(fig_t)

        for fig in fig_list:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        pdf.infodict()["Title"] = "RL Portfolio Optimization v2 — LSTM"

    print(f"PDF saved: {pdf_path}")


# =============================================================================
# MAIN — required for ThreadPoolExecutor / multiprocessing safety on all OSes
# =============================================================================
def main():
    # ── Data & features ──────────────────────────────────────────────────────
    full_prices    = download_data(ASSETS, "2018-01-01", "2024-12-31")
    feature_engine = FeatureEngine(full_prices)

    def get_date_indices(start_str, end_str, min_history=252):
        dates = feature_engine.dates
        mask  = (dates >= start_str) & (dates <= end_str)
        idx   = np.where(mask)[0]
        return idx[idx >= min_history].tolist()

    train_idx = get_date_indices(TRAIN_START, TRAIN_END)
    val_idx   = get_date_indices(VAL_START,   VAL_END)
    test_idx  = get_date_indices(TEST_START,  TEST_END)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} "
          f"| Test: {len(test_idx)} days")

    # ── Sanity check + adaptive ZETA ─────────────────────────────────────────
    print("\n" + "="*60 + "\nSANITY CHECKS\n" + "="*60)
    raw_rets = feature_engine.returns[ASSETS]
    print(f"Max daily return : {raw_rets.max().max()*100:.2f}%")
    print(f"Min daily return : {raw_rets.min().min()*100:.2f}%")
    print(f"Mean daily return: {raw_rets.mean().mean()*100:.4f}%")

    all_flat     = feature_engine._returns_np.flatten()
    cvar_natural = float(np.mean(all_flat[
        all_flat <= np.percentile(all_flat, 5)]))
    global ZETA
    ZETA = abs(cvar_natural) * 0.1
    print(f"Natural CVaR_5%={cvar_natural:.4f} -> ZETA set to {ZETA:.4f}")

    # ── Training ─────────────────────────────────────────────────────────────
    print("\nStarting 4 experiments...")
    results = {}

    results["PPO_CVaR"] = train_ppo(
        "PPO_CVaR", feature_engine, train_idx, use_cvar=True)
    torch.cuda.empty_cache()

    results["PPO_Var"]  = train_ppo(
        "PPO_Var", feature_engine, train_idx, use_cvar=False)
    torch.cuda.empty_cache()

    results["TD3_CVaR"] = train_td3(
        "TD3_CVaR", feature_engine, train_idx, use_cvar=True)
    torch.cuda.empty_cache()

    results["TD3_Var"]  = train_td3(
        "TD3_Var", feature_engine, train_idx, use_cvar=False)
    torch.cuda.empty_cache()

    print("\nAll experiments complete!")

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    for name, res in results.items():
        m = evaluate_policy(res["actor"], feature_engine,
                            val_idx, n_episodes=3)
        print(f"  {name}: CumRet={m['cum_return']:+.2f}% | "
              f"Sharpe={m['sharpe']:.2f} | MDD={m['max_drawdown']:.2f}% | "
              f"CVaR={m['cvar_95']:.4f}")

    print("\nEvaluating on test set (2024)...")
    test_metrics = {}
    for name, res in results.items():
        m = evaluate_policy(res["actor"], feature_engine, test_idx,
                            episode_len=len(test_idx) - 1, n_episodes=1)
        test_metrics[name] = m
        print(f"  {name}: CumRet={m['cum_return']:+.2f}% | "
              f"Sharpe={m['sharpe']:.2f} | MDD={m['max_drawdown']:.2f}% | "
              f"CVaR={m['cvar_95']:.4f} | Turnover={m['turnover']:.4f}")

    baseline = compute_equal_weight_baseline(feature_engine, test_idx)
    test_metrics["EqualWeight"] = baseline
    print(f"  EqualWeight: CumRet={baseline['cum_return']:+.2f}% | "
          f"Sharpe={baseline['sharpe']:.2f} | "
          f"MDD={baseline['max_drawdown']:.2f}%")

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST SET RESULTS (2024)")
    print("="*80)
    print(f"{'Experiment':<20} {'Cum.Return':>12} {'Sharpe':>8} "
          f"{'MaxDrawdown':>13} {'CVaR_0.95':>11} {'Turnover':>10}")
    print("-"*80)
    for name, m in test_metrics.items():
        print(f"{name:<20} {m['cum_return']:>11.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']:>12.2f}% {m['cvar_95']:>11.4f} "
              f"{m['turnover']:>10.4f}")
    print("="*80)

    # ── Plots & PDF ───────────────────────────────────────────────────────────
    fig_list = generate_plots(results, test_metrics)
    save_pdf(fig_list, test_metrics)
    print("Done!")


if __name__ == "__main__":
    main()
