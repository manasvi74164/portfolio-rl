#!/usr/bin/env python3
"""
RL Portfolio Optimization v3 — Phase 1
=======================================
Changes from v2:
  1. Expanded asset universe  : 10 assets (equity + bonds + gold + intl)
  2. Extended date range      : 2000-01-01 to 2024-12-31 (all major regimes)
  3. DCC-GARCH dynamic graph  : time-varying correlation matrix as GAT edges
  4. Asset Graph Attention Net: GAT enriches per-asset embeddings before LSTM
  5. HMM regime posterior     : 3-state HMM appended to static features
  6. PPO + CVaR only          : TD3 and EVaR dropped for clean foundation

Architecture:
  Raw returns/volume (SEQ_LEN, N, 2)
      ↓  per-step
  AssetGAT  [edges = DCC correlation matrix at t]
      ↓
  (SEQ_LEN, GAT_OUT_DIM)  per-asset mean-pool over N assets
      ↓
  LSTM(256)
      ↓
  concat [ LSTM hidden | static(105) ]
      ↓
  Actor MLP → Dirichlet α  (portfolio weights)
  Critic MLP → scalar value

Static feature vector (105-dim):
  prev_weights(10) + cur_weights(10) + cash(1) + constraint_state(1)
  + momentum×volatility indicators(10×4×2=80) + HMM posterior(3)

GPU Optimizations (RTX 4070 + i9):
  - TF32 tensor cores, AMP float16, torch.compile()
  - Pinned-memory rollout buffer, non-blocking GPU transfers
  - DCC correlation tensors pre-staged to GPU
  - N_ENVS=16 threaded vectorised environments
  - MINI_BATCH=256, BATCH_SIZE=2048 for tensor core occupancy
"""

import warnings
warnings.filterwarnings("ignore")

import os, random, math
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
from arch import arch_model
from hmmlearn import hmm as hmmlib
from collections import deque
from typing import List, Tuple, Dict, Optional
import copy
from concurrent.futures import ThreadPoolExecutor

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# HARDWARE SETUP
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark   = True
    torch.backends.cudnn.enabled     = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n" + "!"*60)
    print("WARNING: No CUDA GPU — running on CPU (slow).")
    print("Install CUDA PyTorch:")
    print("  pip install torch torchvision torchaudio "
          "--index-url https://download.pytorch.org/whl/cu124")
    print("!"*60 + "\n")

CPU_CORES = os.cpu_count() or 8
print(f"CPU  : {CPU_CORES} logical cores  |  Device: {DEVICE}")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# ── Asset universe: equity (5) + SPY + bonds + gold + small-cap + intl ───────
ASSETS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
             "SPY",  "TLT",  "GLD",   "IWM",  "EFA"]
N         = len(ASSETS)          # 10
N_ACTIONS = N + 1                # 11  (10 assets + cash)

# ── Date ranges: 25 years → all major regimes ────────────────────────────────
DATA_START  = "2000-01-01"
DATA_END    = "2024-12-31"
TRAIN_START = "2000-01-01";  TRAIN_END = "2019-12-31"
VAL_START   = "2020-01-01";  VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01";  TEST_END  = "2024-12-31"

# ── Network dims ─────────────────────────────────────────────────────────────
SEQ_LEN       = 30
N_RAW_FEAT    = 2                          # returns + volume z-score per asset
GAT_HIDDEN    = 64
GAT_OUT_DIM   = 32                         # per-asset embedding after GAT
GAT_HEADS     = 4
SEQ_FEAT_DIM  = GAT_OUT_DIM               # what LSTM sees per timestep
LSTM_HIDDEN   = 256
LSTM_LAYERS   = 2

N_LOOKBACKS   = 4
N_INDICATORS  = 2
HMM_STATES    = 3

# STATIC_DIM = prev_w(N) + cur_w(N) + cash(1) + cs(1)
#            + indicators(N*N_LOOKBACKS*N_INDICATORS) + hmm(HMM_STATES)
STATIC_DIM = N + N + 1 + 1 + (N * N_LOOKBACKS * N_INDICATORS) + HMM_STATES
# = 10+10+1+1+80+3 = 105

P0            = 100_000.0

# ── Reward / constraint ───────────────────────────────────────────────────────
BETA          = 0.005
XI            = 0.0025
ALPHA_CVAR    = 0.95
ZETA          = 0.005
GAMMA         = 1.0
RHO_0         = 0.001
BETA_RHO      = 1.008
RHO_MAX       = 10.0
LAMBDA_0      = 0.0
ETA_NU        = 1e-3
ETA_LAMBDA    = 1e-3

# ── Training ─────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 500_000
EPISODE_LEN   = 252
LR            = 1e-4
BATCH_SIZE    = 2_048
MINI_BATCH    = 256
N_ENVS        = min(16, max(4, CPU_CORES // 2))
PPO_N_STEPS   = 2_048
PPO_EPOCHS    = 10
PPO_CLIP      = 0.2
PPO_ENT_COEF  = 0.01
KAPPA         = 1.5

# ── GARCH / HMM ──────────────────────────────────────────────────────────────
GARCH_WINDOW  = 252        # rolling window for GARCH fit
DCC_DECAY     = 0.94       # exponential smoothing for DCC quasi-correlation
HMM_WINDOW    = 252        # rolling window for HMM fit
HMM_REFIT     = 63         # refit HMM every 63 trading days
GAT_EDGE_THRESH = 0.05     # ignore edges with |correlation| < threshold

print(f"\nConfig: N={N} | N_ACTIONS={N_ACTIONS} | STATIC_DIM={STATIC_DIM}")
print(f"SEQ_FEAT_DIM={SEQ_FEAT_DIM} | LSTM_HIDDEN={LSTM_HIDDEN}")
print(f"N_ENVS={N_ENVS} | MINI_BATCH={MINI_BATCH} | BATCH_SIZE={BATCH_SIZE}")


# =============================================================================
# DATA DOWNLOAD
# =============================================================================
def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    print(f"\nDownloading {tickers}  {start} → {end} ...")
    raw    = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    # Some tickers (TSLA, GOOGL) listed after 2000 — forward-fill from first valid
    prices = prices.ffill().dropna()
    print(f"  {len(prices)} trading days  shape: {prices.shape}")
    return prices[tickers]   # enforce column order


# =============================================================================
# GARCH ENGINE  —  precomputes (T, N, N) DCC correlation array
# =============================================================================
class GARCHEngine:
    """
    For each asset fits GARCH(1,1) on a rolling GARCH_WINDOW to get
    conditional volatilities σ_i(t).  Standardised residuals are then
    used in a simplified DCC update (exponential smoothing of outer
    products) to produce a time-varying correlation matrix R(t).

    Output stored as:
      self.corr_np   : float32 (T, N, N)   — DCC correlation matrices
      self.cond_vol  : float32 (T, N)       — conditional volatilities
    Both are in pinned CPU memory for fast GPU transfer.
    """

    def __init__(self, returns: pd.DataFrame):
        T, n = returns.shape
        self.T = T
        self.N = n
        rets   = returns.values.astype(np.float64)   # arch needs float64

        print("  Fitting GARCH(1,1) per asset ...")
        std_resid = np.zeros((T, n), dtype=np.float64)
        cond_vol  = np.zeros((T, n), dtype=np.float32)

        for i, col in enumerate(returns.columns):
            r = rets[:, i] * 100          # scale to % for numerical stability
            try:
                res = arch_model(r, vol="Garch", p=1, q=1,
                                 dist="normal", rescale=False).fit(
                                     disp="off", show_warning=False)
                sv             = res.conditional_volatility        # length T
                sv             = np.where(sv < 1e-8, 1e-8, sv)
                std_resid[:, i] = r / sv
                cond_vol[:, i]  = (sv / 100).astype(np.float32)   # back to decimal
            except Exception:
                # fallback: rolling std
                s = pd.Series(r).rolling(21, min_periods=5).std().fillna(1.0).values
                std_resid[:, i] = r / np.where(s < 1e-8, 1e-8, s)
                cond_vol[:, i]  = (s / 100).astype(np.float32)

        print("  Building DCC correlation matrices ...")
        # DCC: Q_t = (1-a)*Q_bar + a * e_{t-1} e_{t-1}' + b * Q_{t-1}
        # We use simplified exponential smoothing (a=1-DCC_DECAY, b=DCC_DECAY)
        Q_bar = np.corrcoef(std_resid.T)
        Q_t   = Q_bar.copy()
        a_dcc = 1.0 - DCC_DECAY

        corr_np = np.zeros((T, n, n), dtype=np.float32)
        for t in range(T):
            if t > 0:
                e = std_resid[t - 1]
                Q_t = (1 - a_dcc) * Q_bar + a_dcc * np.outer(e, e) \
                      + DCC_DECAY * Q_t
            # Rescale Q_t to correlation matrix R_t
            d   = np.sqrt(np.diag(Q_t))
            d   = np.where(d < 1e-8, 1e-8, d)
            R_t = Q_t / np.outer(d, d)
            np.clip(R_t, -1.0, 1.0, out=R_t)
            np.fill_diagonal(R_t, 1.0)
            corr_np[t] = R_t.astype(np.float32)

        # corr_tensor on CPU (pinned) for get_corr() numpy access
        self.corr_tensor = torch.from_numpy(corr_np)
        if DEVICE.type == "cuda":
            self.corr_tensor = self.corr_tensor.pin_memory()

        self.cond_vol = cond_vol         # (T, N) float32 numpy

        # ── Pre-stage full (T, SEQ_LEN, N, N) adjacency tensor to GPU ────
        # For each absolute timestep t, adj_gpu[t] contains the SEQ_LEN
        # correlation matrices ending at t.  ~75 MB on T4/RTX 4070 VRAM.
        # Built once here; all lookups during training are pure GPU indexing.
        print("  Pre-staging adjacency tensor to GPU ...")
        # Build as (T, SEQ_LEN, N, N) by gathering with offset indices
        t_indices = np.arange(T, dtype=np.int64)           # (T,)
        # For each lag s in [0, SEQ_LEN), t_source[s, t] = max(0, t-SEQ_LEN+1+s)
        adj_full = np.zeros((T, SEQ_LEN, N, N), dtype=np.float32)
        for s in range(SEQ_LEN):
            src = np.clip(t_indices - SEQ_LEN + 1 + s, 0, T - 1)
            adj_full[:, s, :, :] = corr_np[src]            # (T, N, N)

        if DEVICE.type == "cuda":
            # Transfer once, keep on GPU permanently
            self.adj_gpu = torch.from_numpy(adj_full).to(DEVICE)
        else:
            self.adj_gpu = torch.from_numpy(adj_full)

        mb = adj_full.nbytes / 1e6
        print(f"  GARCHEngine ready: corr {corr_np.shape} | "
              f"adj_gpu {tuple(self.adj_gpu.shape)} ({mb:.1f} MB on {DEVICE})")

    def get_corr(self, t: int) -> np.ndarray:
        """Returns (N, N) float32 correlation matrix at time t."""
        return self.corr_tensor[t].numpy()

    def get_corr_batch(self, t_indices: np.ndarray) -> torch.Tensor:
        """Returns (B, N, N) tensor for a batch of timesteps."""
        return self.corr_tensor[t_indices]


# =============================================================================
# HMM ENGINE  —  precomputes (T, 3) regime posterior array
# =============================================================================
class HMMEngine:
    """
    Fits a 3-state Gaussian HMM on a multivariate observation vector:
      [realized_vol, avg_pairwise_corr, momentum_dispersion]
    Refits every HMM_REFIT days on a rolling HMM_WINDOW.

    Output:
      self.posterior_np : float32 (T, HMM_STATES)
    States are sorted by mean realized vol: state 0 = bull (low vol),
    state 1 = bear (mid vol), state 2 = crisis (high vol).
    """

    def __init__(self, returns: pd.DataFrame, corr_engine: GARCHEngine):
        T, n  = returns.shape
        rets  = returns.values.astype(np.float64)

        print("  Building HMM observation series ...")
        obs_seq = np.zeros((T, 3), dtype=np.float64)

        for t in range(T):
            window    = max(0, t - 21)
            r_win     = rets[window:t + 1]
            rv        = float(np.std(r_win)) * np.sqrt(252) if len(r_win) > 1 else 0.0
            corr_mat  = corr_engine.get_corr(t)
            mask      = np.triu(np.ones((n, n), dtype=bool), k=1)
            avg_corr  = float(np.mean(np.abs(corr_mat[mask])))
            mom       = float(np.std(rets[t]))   # cross-sectional dispersion
            obs_seq[t] = [rv, avg_corr, mom]

        # Normalise each column to zero mean unit variance
        mu_obs  = obs_seq.mean(axis=0)
        sd_obs  = obs_seq.std(axis=0) + 1e-8
        obs_seq = (obs_seq - mu_obs) / sd_obs

        print("  Fitting rolling HMM ...")
        posterior_np = np.zeros((T, HMM_STATES), dtype=np.float32)
        model        = None

        for t in range(T):
            # Refit at t=0 and every HMM_REFIT days thereafter
            if t % HMM_REFIT == 0:
                w_start = max(0, t - HMM_WINDOW)
                X_fit   = obs_seq[w_start:t + 1]
                if len(X_fit) >= HMM_STATES * 10:
                    try:
                        m = hmmlib.GaussianHMM(
                            n_components=HMM_STATES,
                            covariance_type="full",
                            n_iter=100,
                            random_state=SEED)
                        m.fit(X_fit)
                        # Sort states by mean realised vol (component 0)
                        order = np.argsort(m.means_[:, 0])
                        m.means_       = m.means_[order]
                        m.covars_      = m.covars_[order]
                        m.startprob_   = m.startprob_[order]
                        m.transmat_    = m.transmat_[order][:, order]
                        model = m
                    except Exception:
                        pass

            if model is not None:
                try:
                    _, post = model.score_samples(obs_seq[max(0, t-1):t + 1])
                    posterior_np[t] = post[-1].astype(np.float32)
                except Exception:
                    posterior_np[t] = np.ones(HMM_STATES, dtype=np.float32) / HMM_STATES
            else:
                posterior_np[t] = np.ones(HMM_STATES, dtype=np.float32) / HMM_STATES

        self.posterior_np = posterior_np   # (T, HMM_STATES) float32
        print(f"  HMMEngine ready: posterior shape {posterior_np.shape}")

    def get_posterior(self, t: int) -> np.ndarray:
        return self.posterior_np[t]


# =============================================================================
# FEATURE ENGINE  —  integrates GARCH + HMM outputs
# =============================================================================
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)

def rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu  = df.rolling(window).mean()
    sig = df.rolling(window).std()
    return ((df - mu) / (sig + 1e-8)).fillna(0.0)

def compute_momentum(prices: pd.DataFrame, L: int) -> pd.DataFrame:
    return ((prices - prices.shift(L)) / (prices.shift(L) + 1e-8)).fillna(0.0)

def compute_volatility(returns: pd.DataFrame, L: int) -> pd.DataFrame:
    return (returns.rolling(L).std() * np.sqrt(252)).fillna(0.0)


class FeatureEngine:
    """
    Precomputes ALL features once as contiguous float32 numpy arrays.
    get_state() returns:
      seq_raw   : (SEQ_LEN, N, N_RAW_FEAT)   — raw per-asset features for GAT
      t_idx_val : int                          — timestep (for GAT corr lookup)
      static    : (STATIC_DIM,)               — static snapshot
    """

    def __init__(self, prices: pd.DataFrame):
        print("\nBuilding FeatureEngine ...")
        self.prices  = prices
        self.returns = compute_returns(prices)

        # Volume
        try:
            vol_raw = yf.download(ASSETS, start=DATA_START, end=DATA_END,
                                  auto_adjust=True, progress=False)["Volume"]
            self.volume_zscored = rolling_zscore(
                vol_raw.reindex(prices.index).fillna(0), 30)
        except Exception:
            self.volume_zscored = pd.DataFrame(
                np.zeros((len(prices), N), dtype=np.float32),
                index=prices.index, columns=ASSETS)

        # Momentum & volatility indicators
        self.lookbacks  = [21, 63, 126, 252]
        self.momentum   = {L: compute_momentum(prices, L)
                           for L in self.lookbacks}
        self.volatility = {L: compute_volatility(self.returns, L)
                           for L in self.lookbacks}

        # Cache as float32 numpy
        self._returns_np = self.returns.values.astype(np.float32)
        self._volume_np  = rolling_zscore(self.volume_zscored, 30
                           ).values.astype(np.float32)
        self._mom_np = {L: self.momentum[L].values.astype(np.float32)
                        for L in self.lookbacks}
        self._vol_np = {L: self.volatility[L].values.astype(np.float32)
                        for L in self.lookbacks}

        self.dates  = prices.index
        self.n_days = len(prices)

        # GARCH + DCC
        print("Building GARCHEngine ...")
        self.garch = GARCHEngine(self.returns)

        # HMM
        print("Building HMMEngine ...")
        self.hmm = HMMEngine(self.returns, self.garch)

        print(f"FeatureEngine ready: {self.n_days} days, "
              f"STATIC_DIM={STATIC_DIM}")

    def get_state(self, t: int,
                  prev_weights: np.ndarray,
                  current_weights: np.ndarray,
                  cash_balance: float,
                  constraint_state: float
                  ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Returns
        -------
        seq_raw  : (SEQ_LEN, N, N_RAW_FEAT)   float32
        t_val    : int   — absolute timestep for GAT corr lookup
        static   : (STATIC_DIM,)               float32
        """
        t_start    = max(0, t - SEQ_LEN + 1)
        ret_window = self._returns_np[t_start:t + 1]       # (≤SEQ_LEN, N)
        vol_window = self._volume_np[t_start:t + 1]

        if ret_window.shape[0] < SEQ_LEN:
            pad        = SEQ_LEN - ret_window.shape[0]
            ret_window = np.vstack([np.zeros((pad, N), np.float32), ret_window])
            vol_window = np.vstack([np.zeros((pad, N), np.float32), vol_window])

        ret_window = np.clip(ret_window, -5.0, 5.0)
        vol_window = np.clip(vol_window, -5.0, 5.0)

        # seq_raw: (SEQ_LEN, N, 2)  — stacked along last axis
        seq_raw = np.stack([ret_window, vol_window], axis=-1).astype(np.float32)

        # Static features
        prev_w = prev_weights.astype(np.float32)
        cur_w  = current_weights.astype(np.float32)
        cash_n = np.array([cash_balance / P0], dtype=np.float32)
        cs     = np.array([np.clip(constraint_state, -10, 10)], dtype=np.float32)

        ind_list = []
        for L in self.lookbacks:
            ind_list.extend([
                np.clip(self._mom_np[L][t], -2, 2),
                np.clip(self._vol_np[L][t],  0, 5),
            ])

        regime = self.hmm.get_posterior(t)   # (3,) float32

        static = np.concatenate([prev_w, cur_w, cash_n, cs,
                                  np.concatenate(ind_list), regime])
        assert static.shape[0] == STATIC_DIM, \
            f"STATIC_DIM mismatch: {static.shape[0]} vs {STATIC_DIM}"
        return seq_raw, t, static


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
        self.portfolio_value           = P0
        self.cash_balance              = P0
        self.weights                   = np.zeros(N_ACTIONS, dtype=np.float32)
        self.weights[-1]               = 1.0
        self.prev_weights              = self.weights.copy()
        self.constraint_state          = 0.0
        self.portfolio_returns_history = []
        self.nu          = 0.0
        self.lambda_cvar = 0.0
        self.rho_alm     = RHO_0

    def reset(self) -> Tuple[np.ndarray, int, np.ndarray]:
        self._reset_internal()
        max_start = len(self.date_indices) - self.episode_len - 1
        start_pos = random.randint(0, max(0, max_start))
        self.t_idx  = self.date_indices[start_pos]
        self.t_step = 0
        return self._get_state()

    def _get_state(self) -> Tuple[np.ndarray, int, np.ndarray]:
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
            -1.0, 1.0))

        self.constraint_state  = 0.99 * self.constraint_state + C_estimate
        self.prev_weights      = self.weights.copy()
        self.weights           = action.copy()
        self.portfolio_value  *= (1.0 + r_portfolio)
        self.cash_balance      = self.portfolio_value * action[-1]
        self.t_idx            += 1
        self.t_step           += 1
        done = (self.t_step >= self.episode_len)

        info = {"r_portfolio": r_portfolio, "sigma_sq": sigma_sq,
                "turnover": turnover, "C_estimate": C_estimate,
                "portfolio_value": self.portfolio_value}
        return self._get_state(), reward, done, info

    def set_alm_params(self, nu: float, lambda_cvar: float, rho_alm: float):
        self.nu, self.lambda_cvar, self.rho_alm = nu, lambda_cvar, rho_alm


# =============================================================================
# THREADED VECTORISED ENV
# State is now (seq_raw, t_idx, static) — 3-tuple instead of 2-tuple
# =============================================================================
class ThreadedVecEnv:
    def __init__(self, env_fns):
        self.envs     = [fn() for fn in env_fns]
        self.n        = len(self.envs)
        self.executor = ThreadPoolExecutor(max_workers=self.n)

    def _pack(self, states):
        seqs    = np.stack([s[0] for s in states])   # (E, SEQ_LEN, N, 2)
        t_idxs  = np.array([s[1] for s in states],
                            dtype=np.int64)            # (E,)
        statics = np.stack([s[2] for s in states])   # (E, STATIC_DIM)
        return seqs, t_idxs, statics

    def reset(self):
        futs   = [self.executor.submit(e.reset) for e in self.envs]
        return self._pack([f.result() for f in futs])

    def step(self, actions: np.ndarray):
        futs    = [self.executor.submit(e.step, a)
                   for e, a in zip(self.envs, actions)]
        results = [f.result() for f in futs]
        next_states, rewards, dones, infos = zip(*results)

        seqs, t_idxs, statics = self._pack(next_states)
        rewards = np.array(rewards, dtype=np.float32)
        dones   = np.array(dones,   dtype=bool)

        # Auto-reset done envs
        reset_futs = {i: self.executor.submit(self.envs[i].reset)
                      for i, d in enumerate(dones) if d}
        for i, fut in reset_futs.items():
            rs, rt, rstat = fut.result()
            seqs[i]    = rs
            t_idxs[i]  = rt
            statics[i] = rstat

        return (seqs, t_idxs, statics), rewards, dones, list(infos)

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
# CONSTRAINT MANAGERS  (unchanged from v2)
# =============================================================================
class CVaRManager:
    def __init__(self, alpha=ALPHA_CVAR, zeta=ZETA, device=DEVICE):
        self.alpha    = alpha
        self.zeta     = zeta
        self.device   = device
        self.nu       = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.lam      = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.z_buffer = deque(maxlen=100)

    def trajectory_cost(self, episode_returns: List[float]) -> float:
        n   = len(episode_returns)
        cum = float(np.prod([1 + r for r in episode_returns]) - 1)
        Z   = -((1 + cum) ** (252 / max(n, 1)) - 1) if n > 0 else 0.0
        self.z_buffer.append(Z)
        return Z

    def compute_H(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z      = (torch.tensor([0.0], device=self.device)
                  if not self.z_buffer
                  else torch.tensor(list(self.z_buffer),
                                    device=self.device, dtype=torch.float32))
        excess = torch.relu(z - self.nu)
        H      = self.nu + excess.mean() / (1.0 - self.alpha)
        return H, H - self.zeta

    def update(self):
        if len(self.z_buffer) < 2:
            return
        Z        = torch.tensor(list(self.z_buffer),
                                device=self.device, dtype=torch.float32)
        p_exceed = (Z >= self.nu).float().mean()
        self.nu  = self.nu - ETA_NU * (1.0 - p_exceed / (1.0 - self.alpha))
        _, C     = self.compute_H()
        self.lam = torch.clamp(self.lam + ETA_LAMBDA * C, min=0.0)

    @property
    def nu_val(self):  return float(self.nu.item())
    @property
    def lam_val(self): return float(self.lam.item())


class ALMManager:
    def __init__(self, rho0=RHO_0, beta_rho=BETA_RHO, lambda0=LAMBDA_0):
        self.rho = rho0; self.beta_rho = beta_rho; self.lam = lambda0

    def alm_loss(self, C: torch.Tensor) -> torch.Tensor:
        return self.lam * C + (self.rho / 2.0) * C ** 2

    def update(self, C: float):
        self.lam = max(0.0, self.lam + self.rho * C)
        if C > 0:
            self.rho = min(self.beta_rho * self.rho, RHO_MAX)

    @property
    def lam_val(self): return self.lam
    @property
    def rho_val(self): return self.rho


# =============================================================================
# ASSET GRAPH ATTENTION NETWORK (native PyTorch — no torch_geometric needed)
# =============================================================================
class AssetGAT(nn.Module):
    """
    Multi-head Graph Attention over N assets.

    Input:
      x    : (B, N, in_dim)     — per-asset node features
      adj  : (B, N, N)          — DCC correlation matrix (edge weights)

    Output:
      out  : (B, GAT_OUT_DIM)   — mean-pooled asset embedding

    Architecture:
      Layer 1: GAT_HEADS attention heads, hidden = GAT_HIDDEN
      Layer 2: single-head projection to GAT_OUT_DIM
    """

    def __init__(self, in_dim: int = N_RAW_FEAT * SEQ_LEN,
                 hidden: int = GAT_HIDDEN,
                 out_dim: int = GAT_OUT_DIM,
                 heads: int = GAT_HEADS):
        super().__init__()
        self.heads   = heads
        self.h_size  = hidden // heads   # per-head hidden dim
        self.out_dim = out_dim

        # Layer 1 — multi-head attention
        self.W_src  = nn.Linear(in_dim,  self.h_size * heads, bias=False)
        self.W_dst  = nn.Linear(in_dim,  self.h_size * heads, bias=False)
        self.att1   = nn.Linear(2 * self.h_size * heads, 1, bias=False)
        self.W_val1 = nn.Linear(in_dim, hidden, bias=False)
        self.norm1  = nn.LayerNorm(hidden)

        # Layer 2 — single-head projection
        self.W_src2 = nn.Linear(hidden, out_dim, bias=False)
        self.W_dst2 = nn.Linear(hidden, out_dim, bias=False)
        self.att2   = nn.Linear(2 * out_dim, 1, bias=False)
        self.W_val2 = nn.Linear(hidden, out_dim, bias=False)
        self.norm2  = nn.LayerNorm(out_dim)

    def _gat_layer(self, x, adj, W_src, W_dst, att_fc, W_val, norm):
        """
        x   : (B, N, d_in)
        adj : (B, N, N)   edge weights (absolute correlation)
        """
        B, n_nodes, _ = x.shape
        h_src  = W_src(x)                          # (B, N, d_out)
        h_dst  = W_dst(x)                          # (B, N, d_out)
        d_out  = h_src.shape[-1]

        # Attention logits: for each (i,j) pair
        # e_ij = LeakyReLU( att( [h_i || h_j] ) )
        h_i    = h_src.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (B,N,N,d)
        h_j    = h_dst.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (B,N,N,d)
        e      = F.leaky_relu(att_fc(torch.cat([h_i, h_j], dim=-1)).squeeze(-1),
                              negative_slope=0.2)                  # (B,N,N)

        # Mask near-zero edges and apply DCC weights
        mask   = (adj.abs() < GAT_EDGE_THRESH)
        e      = e + adj * 0.5                    # incorporate DCC weight
        e      = e.masked_fill(mask, float("-inf"))

        alpha  = torch.softmax(e, dim=-1)          # (B, N, N)
        alpha  = torch.nan_to_num(alpha, nan=1.0 / n_nodes)

        # Aggregate
        v      = W_val(x)                          # (B, N, d_out)
        out    = torch.bmm(alpha, v)               # (B, N, d_out)
        out    = norm(F.elu(out))
        return out

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N, in_dim)
        adj : (B, N, N)
        returns: (B, GAT_OUT_DIM)
        """
        x   = torch.nan_to_num(x,   nan=0.0, posinf=5.0, neginf=-5.0)
        adj = torch.nan_to_num(adj, nan=0.0)

        h   = self._gat_layer(x, adj, self.W_src,  self.W_dst,
                               self.att1, self.W_val1, self.norm1)
        out = self._gat_layer(h, adj, self.W_src2, self.W_dst2,
                               self.att2, self.W_val2, self.norm2)
        return out.mean(dim=1)   # (B, GAT_OUT_DIM)  — mean pool over assets


# =============================================================================
# NETWORK BACKBONE  —  GAT → LSTM → MLP
# =============================================================================
class PortfolioBackbone(nn.Module):
    """
    Shared feature extractor used by both actor and critic.
    Input per timestep:
      seq_raw  : (B, SEQ_LEN, N, N_RAW_FEAT)  — raw OHLCV-style features
      adj_seq  : (B, SEQ_LEN, N, N)            — DCC corr for each step in window
      static   : (B, STATIC_DIM)

    Processing:
      For each of SEQ_LEN timesteps: AssetGAT(raw_t, adj_t) → (B, GAT_OUT_DIM)
      Stack → (B, SEQ_LEN, GAT_OUT_DIM)
      LSTM  → (B, LSTM_HIDDEN)
      concat static → (B, LSTM_HIDDEN + STATIC_DIM)
    """

    def __init__(self):
        super().__init__()
        self.gat  = AssetGAT(in_dim=N_RAW_FEAT, hidden=GAT_HIDDEN,
                              out_dim=GAT_OUT_DIM, heads=GAT_HEADS)
        self.lstm = nn.LSTM(input_size=GAT_OUT_DIM,
                            hidden_size=LSTM_HIDDEN,
                            num_layers=LSTM_LAYERS,
                            batch_first=True,
                            dropout=0.1 if LSTM_LAYERS > 1 else 0.0)
        self.out_dim = LSTM_HIDDEN + STATIC_DIM

    def forward(self, seq_raw: torch.Tensor,
                adj_seq: torch.Tensor,
                static:  torch.Tensor) -> torch.Tensor:
        """
        seq_raw : (B, SEQ_LEN, N, N_RAW_FEAT)
        adj_seq : (B, SEQ_LEN, N, N)
        static  : (B, STATIC_DIM)
        returns : (B, LSTM_HIDDEN + STATIC_DIM)
        """
        B, T, n_nodes, f = seq_raw.shape

        # Run GAT independently per timestep
        # Reshape to (B*T, N, f) and (B*T, N, N) for batched GAT
        x_flat   = seq_raw.reshape(B * T, n_nodes, f)
        adj_flat = adj_seq.reshape(B * T, n_nodes, n_nodes)
        gat_out  = self.gat(x_flat, adj_flat)          # (B*T, GAT_OUT_DIM)
        gat_seq  = gat_out.reshape(B, T, GAT_OUT_DIM)  # (B, T, GAT_OUT_DIM)

        # LSTM
        _, (h_n, _) = self.lstm(gat_seq)
        h           = h_n[-1]                           # (B, LSTM_HIDDEN)

        static = torch.nan_to_num(static, nan=0.0, posinf=5.0, neginf=-5.0)
        return torch.cat([h, static], dim=-1)           # (B, LSTM_HIDDEN+STATIC_DIM)


class LSTMActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PortfolioBackbone()
        d = self.backbone.out_dim
        self.head = nn.Sequential(
            nn.Linear(d, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, N_ACTIONS))

    def forward(self, seq_raw, adj_seq, static):
        feat   = self.backbone(seq_raw, adj_seq, static)
        logits = torch.clamp(self.head(feat), -10.0, 10.0)
        alpha  = torch.nan_to_num(F.softplus(logits) + 1e-6, nan=1.0)
        return torch.clamp(alpha, min=1e-3, max=1e3)

    def get_action(self, seq_raw, adj_seq, static, explore=True):
        alpha  = self.forward(seq_raw, adj_seq, static)
        dist   = Dirichlet(torch.clamp(KAPPA * alpha if explore else alpha,
                                       min=1e-3, max=1e3))
        action   = dist.rsample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def get_deterministic_action(self, seq_raw, adj_seq, static):
        alpha = self.forward(seq_raw, adj_seq, static)
        return alpha / alpha.sum(dim=-1, keepdim=True)


class LSTMValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PortfolioBackbone()
        d = self.backbone.out_dim
        self.head = nn.Sequential(
            nn.Linear(d, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 1))

    def forward(self, seq_raw, adj_seq, static):
        return self.head(self.backbone(seq_raw, adj_seq, static))


def maybe_compile(model: nn.Module) -> nn.Module:
    import sys
    if sys.platform == "win32":
        print(f"  torch.compile() skipped (Windows)")
        return model
    if int(torch.__version__.split(".")[0]) < 2:
        return model
    try:
        c = torch.compile(model, mode="reduce-overhead")
        print(f"  torch.compile() OK: {type(model).__name__}")
        return c
    except Exception as e:
        print(f"  torch.compile() skipped ({type(e).__name__})")
    return model


# =============================================================================
# ROLLOUT BUFFER  —  extended for (seq_raw, t_idx, static) state format
# =============================================================================
class RolloutBuffer:
    """
    Pre-allocated pinned-memory tensors.
    Stores seq_raw (SEQ_LEN, N, N_RAW_FEAT) + t_idx (int) + static per step.
    adj_seq is looked up from garch engine at batch-generation time.
    """

    def __init__(self, n_steps, n_envs, device=DEVICE):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.device  = device
        pin = {"pin_memory": True} if device.type == "cuda" else {}

        self.seqs      = torch.zeros(n_steps, n_envs,
                                     SEQ_LEN, N, N_RAW_FEAT, **pin)
        self.t_idxs    = torch.zeros(n_steps, n_envs, dtype=torch.int64)
        self.statics   = torch.zeros(n_steps, n_envs, STATIC_DIM, **pin)
        self.actions   = torch.zeros(n_steps, n_envs, N_ACTIONS, **pin)
        self.rewards   = torch.zeros(n_steps, n_envs, **pin)
        self.log_probs = torch.zeros(n_steps, n_envs, **pin)
        self.values    = torch.zeros(n_steps, n_envs, **pin)
        self.dones     = torch.zeros(n_steps, n_envs, **pin)
        self.ptr = 0

    def add(self, seq, t_idx, static, action, reward, log_prob, value, done):
        i = self.ptr
        self.seqs[i].copy_(torch.from_numpy(seq))
        self.t_idxs[i].copy_(torch.from_numpy(t_idx))
        self.statics[i].copy_(torch.from_numpy(static))
        self.actions[i].copy_(torch.from_numpy(action))
        self.rewards[i]   = torch.from_numpy(reward)
        self.log_probs[i] = torch.from_numpy(log_prob)
        self.values[i]    = torch.from_numpy(value)
        self.dones[i]     = torch.from_numpy(done)
        self.ptr = (self.ptr + 1) % self.n_steps

    def get_batches(self, batch_size, last_values,
                    garch_engine: GARCHEngine, gamma=GAMMA):
        T, E       = self.n_steps, self.n_envs
        gae_lambda = 0.95
        values_np  = self.values.numpy()
        rewards_np = self.rewards.numpy()
        dones_np   = self.dones.numpy()

        # GAE
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

        # Flatten everything
        flat_seqs   = self.seqs.view(-1, SEQ_LEN, N, N_RAW_FEAT)
        flat_tidxs  = self.t_idxs.view(-1).numpy()               # (T*E,) int64
        flat_stats  = self.statics.view(-1, STATIC_DIM)
        flat_acts   = self.actions.view(-1, N_ACTIONS)
        flat_rets   = torch.from_numpy(returns.reshape(-1))
        flat_advs   = torch.from_numpy(advantages.reshape(-1))
        flat_lps    = self.log_probs.view(-1)

        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)

        # adj_seq — index directly into pre-staged GPU tensor, zero CPU work
        # flat_tidxs is (T*E,) int64; adj_gpu[flat_tidxs] → (T*E, SEQ_LEN, N, N)
        n_flat    = flat_seqs.shape[0]
        tidx_gpu  = torch.from_numpy(flat_tidxs).to(self.device,
                                                     non_blocking=True)
        flat_adjs = garch_engine.adj_gpu[tidx_gpu]   # (T*E, SEQ_LEN, N, N)

        idx     = torch.randperm(n_flat)               # CPU index
        idx_gpu = idx.to(self.device, non_blocking=True)
        for start in range(0, n_flat, batch_size):
            b     = idx[start:start + batch_size]      # CPU slice → indexes CPU tensors
            b_gpu = idx_gpu[start:start + batch_size]  # GPU slice → indexes GPU tensors
            yield (
                flat_seqs[b].to(self.device, non_blocking=True),
                flat_adjs[b_gpu],                      # already on GPU
                flat_stats[b].to(self.device, non_blocking=True),
                flat_acts[b].to(self.device, non_blocking=True),
                flat_rets[b].to(self.device, non_blocking=True),
                flat_advs[b].to(self.device, non_blocking=True),
                flat_lps[b].to(self.device, non_blocking=True),
            )


# =============================================================================
# STATE → TENSORS HELPER
# =============================================================================
def state_to_tensors(seqs_np, t_idxs_np, statics_np,
                     garch_engine: GARCHEngine):
    """
    Converts numpy state arrays to GPU tensors.
    seqs_np   : (B, SEQ_LEN, N, N_RAW_FEAT)
    t_idxs_np : (B,)  int64
    statics_np: (B, STATIC_DIM)
    adj_t is fetched by indexing the pre-staged GPU tensor — zero CPU work.
    Returns seq_t, adj_t, stat_t all on DEVICE.
    """
    idx_t  = torch.from_numpy(t_idxs_np.astype(np.int64))
    adj_t  = garch_engine.adj_gpu[idx_t]                   # (B, SEQ_LEN, N, N)
    seq_t  = torch.from_numpy(seqs_np).to(DEVICE, non_blocking=True)
    stat_t = torch.from_numpy(statics_np).to(DEVICE, non_blocking=True)
    return seq_t, adj_t, stat_t


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
# PPO TRAINING  —  PPO + CVaR only
# =============================================================================
def train_ppo(experiment_name: str,
              feature_engine: FeatureEngine,
              train_idx: List[int],
              total_steps: int = TOTAL_STEPS) -> Dict:

    print(f"\n{'='*65}")
    print(f"  Training PPO + CVaR  |  {experiment_name}")
    print(f"{'='*65}")

    garch_engine = feature_engine.garch

    actor  = maybe_compile(LSTMActorNetwork().to(DEVICE))
    critic = maybe_compile(LSTMValueNetwork().to(DEVICE))
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    cvar_mgr = CVaRManager()
    alm_mgr  = ALMManager()

    env_fns = [make_env(train_idx, feature_engine) for _ in range(N_ENVS)]
    vec_env = ThreadedVecEnv(env_fns)
    buf     = RolloutBuffer(PPO_N_STEPS, N_ENVS)

    logs = {k: [] for k in ["episode_returns", "cvar_vals", "violations",
                              "lambdas", "rhos", "sigma_sqs",
                              "turnovers", "steps"]}

    obs_seqs, obs_tidxs, obs_statics = vec_env.reset()
    ep_rets_per_env = [[] for _ in range(N_ENVS)]
    step, ep_count  = 0, 0
    scaler          = amp.GradScaler()

    while step < total_steps:
        buf.ptr = 0

        # ── Rollout collection ────────────────────────────────────────────
        for _ in range(PPO_N_STEPS):
            seq_t, adj_t, stat_t = state_to_tensors(
                obs_seqs, obs_tidxs, obs_statics, garch_engine)

            with torch.no_grad(), amp.autocast():
                alpha    = actor(seq_t, adj_t, stat_t)
                dist     = Dirichlet(alpha)
                actions  = dist.rsample()
                log_prob = dist.log_prob(actions)
                values   = critic(seq_t, adj_t, stat_t).squeeze(-1)

            actions_np  = actions.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            values_np   = values.cpu().numpy()

            (next_seqs, next_tidxs, next_statics), rewards, dones, infos = \
                vec_env.step(actions_np)

            buf.add(obs_seqs, obs_tidxs, obs_statics,
                    actions_np, rewards, log_prob_np,
                    values_np, dones.astype(np.float32))

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

                    cum_ret = annualized_return(ep_ret)
                    logs["episode_returns"].append(cum_ret)
                    logs["cvar_vals"].append(
                        float(cvar_mgr.compute_H()[0].item()))
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
                              f"λ={alm_mgr.lam_val:.4f} | "
                              f"ρ={alm_mgr.rho_val:.4f} | C={C_val:.4f}")

            obs_seqs, obs_tidxs, obs_statics = next_seqs, next_tidxs, next_statics
            step += N_ENVS

        # ── Bootstrap value for GAE ───────────────────────────────────────
        with torch.no_grad(), amp.autocast():
            seq_t, adj_t, stat_t = state_to_tensors(
                obs_seqs, obs_tidxs, obs_statics, garch_engine)
            lv = critic(seq_t, adj_t, stat_t).squeeze(-1).cpu().numpy()

        # ── PPO update epochs ─────────────────────────────────────────────
        for _ in range(PPO_EPOCHS):
            for (s, adj, st, a, ret, adv, old_lp) in \
                    buf.get_batches(MINI_BATCH, lv, garch_engine):

                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)

                with amp.autocast():
                    alpha_  = actor(s, adj, st)
                    dist_   = Dirichlet(alpha_)
                    new_lp  = dist_.log_prob(a)
                    entropy = dist_.entropy()
                    ratio   = torch.exp(new_lp - old_lp)
                    clip_r  = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                    a_loss  = -torch.min(ratio * adv, clip_r * adv).mean()
                    a_loss -= PPO_ENT_COEF * entropy.mean()

                    _, C_t  = cvar_mgr.compute_H()
                    a_loss += alm_mgr.alm_loss(C_t)

                    v_loss  = F.mse_loss(critic(s, adj, st).squeeze(-1), ret)

                scaler.scale(a_loss).backward()
                scaler.unscale_(actor_opt)
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                scaler.step(actor_opt)

                scaler.scale(v_loss).backward()
                scaler.unscale_(critic_opt)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                scaler.step(critic_opt)

                scaler.update()

    print(f"\nPPO done. Total episodes: {ep_count}")
    return {"actor": actor, "critic": critic, "logs": logs,
            "cvar_mgr": cvar_mgr, "alm_mgr": alm_mgr,
            "experiment": experiment_name}


# =============================================================================
# EVALUATION
# =============================================================================
def compute_sharpe(returns, rf=0.05 / 252):
    r  = np.array(returns)
    ex = r - rf
    return float(np.mean(ex) / (np.std(ex) + 1e-8) * np.sqrt(252))

def compute_max_drawdown(returns):
    cum  = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-8)
    return float(np.min(dd) * 100)

def compute_cvar(returns, alpha=ALPHA_CVAR):
    r   = np.array(returns)
    var = np.percentile(r, (1 - alpha) * 100)
    tail = r[r <= var]
    return float(np.mean(tail)) if len(tail) > 0 else float(var)


def evaluate_policy(actor, feature_engine: FeatureEngine,
                    date_indices, episode_len=None, n_episodes=3):
    actor.eval()
    garch_engine = feature_engine.garch
    episode_len  = episode_len or min(EPISODE_LEN, len(date_indices) - 1)
    all_rets, all_turns = [], []

    for _ in range(n_episodes):
        env = PortfolioEnv(feature_engine, date_indices,
                           episode_len=episode_len, mode="eval")
        obs = env.reset()   # (seq, t_idx, static)
        ep_rets, ep_turns = [], []

        for _ in range(episode_len):
            seq_np  = obs[0][np.newaxis]   # (1, SEQ_LEN, N, N_RAW_FEAT)
            tidx_np = np.array([obs[1]], dtype=np.int64)
            stat_np = obs[2][np.newaxis]

            seq_t, adj_t, stat_t = state_to_tensors(
                seq_np, tidx_np, stat_np, garch_engine)

            with torch.no_grad(), amp.autocast():
                action = actor.get_deterministic_action(seq_t, adj_t, stat_t)

            obs, _, done, info = env.step(action.squeeze(0).cpu().numpy())
            ep_rets.append(info.get("r_portfolio", 0.0))
            ep_turns.append(info.get("turnover", 0.0))
            if done:
                break

        all_rets.append(ep_rets)
        all_turns.append(float(np.mean(ep_turns)))

    actor.train()
    rets    = all_rets[0]
    cum_ret = float(np.prod([1 + r for r in rets]) - 1) * 100
    return {"cum_return": cum_ret,
            "sharpe":       compute_sharpe(rets),
            "max_drawdown": compute_max_drawdown(rets),
            "cvar_95":      compute_cvar(rets),
            "turnover":     float(np.mean(all_turns)),
            "returns":      rets}


def compute_equal_weight_baseline(feature_engine: FeatureEngine,
                                   date_indices):
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
COLORS  = {"PPO_CVaR": "#1f77b4", "EqualWeight": "#9467bd"}
LSTYLES = {"PPO_CVaR": "-",       "EqualWeight": "--"}


def generate_plots(results: Dict, test_metrics: Dict) -> List:
    fig_list = []

    # ── Fig 1: Cumulative returns on test set ─────────────────────────────
    fig1, ax = plt.subplots(figsize=(13, 6))
    for name, m in test_metrics.items():
        cum = np.cumprod(1 + np.array(m["returns"])) * 100 - 100
        ax.plot(cum, label=name,
                color=COLORS.get(name, "gray"),
                linestyle=LSTYLES.get(name, "-"), lw=2)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title("Cumulative Return — Test Set (2023–2024)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig_list.append(fig1)

    # ── Fig 2: Training curve ─────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(13, 5))
    ep   = results["PPO_CVaR"]["logs"]["episode_returns"]
    ax.plot(ep, alpha=0.4, lw=0.8, color="#1f77b4", label="Per-episode")
    w  = max(1, len(ep) // 20)
    ax.plot(pd.Series(ep).rolling(w, min_periods=1).mean(),
            color="black", lw=1.8, label=f"Rolling mean (w={w})")
    ax.set_title("PPO_CVaR Training Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Annualised Return (%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig_list.append(fig2)

    # ── Fig 3: CVaR constraint tracking ──────────────────────────────────
    fig3, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()
    viol = results["PPO_CVaR"]["logs"]["violations"]
    lams = results["PPO_CVaR"]["logs"]["lambdas"]
    ax1.plot(viol, alpha=0.7, color="tab:red",  label="C (violation)")
    ax2.plot(lams, alpha=0.7, color="tab:blue",
             ls="--", label="λ (dual var)")
    ax1.set_title("CVaR Constraint — Violation & Dual Variable",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("C", color="tab:red")
    ax2.set_ylabel("λ", color="tab:blue")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(); fig_list.append(fig3)

    # ── Fig 4: Regime posterior over test period ──────────────────────────
    # Pull HMM posteriors for the test date indices (already computed)
    fig4, ax = plt.subplots(figsize=(13, 4))
    regime_labels = ["Bull (low vol)", "Bear (mid vol)", "Crisis (high vol)"]
    regime_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    # We need date indices for test period from results
    if "test_posteriors" in results["PPO_CVaR"]:
        post = results["PPO_CVaR"]["test_posteriors"]
        for k in range(HMM_STATES):
            ax.fill_between(range(len(post)), post[:, k],
                            alpha=0.55, color=regime_colors[k],
                            label=regime_labels[k])
    ax.set_title("HMM Regime Posterior — Test Set",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Posterior Probability")
    ax.set_ylim(0, 1); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig_list.append(fig4)

    # ── Fig 5: Test metrics bar chart ─────────────────────────────────────
    fig5, axes = plt.subplots(1, 3, figsize=(14, 5))
    names  = list(test_metrics.keys())
    colors = [COLORS.get(n, "#8c564b") for n in names]
    for ax, vals, title in zip(
            axes,
            [[test_metrics[n]["sharpe"]       for n in names],
             [abs(test_metrics[n]["max_drawdown"]) for n in names],
             [test_metrics[n]["turnover"]      for n in names]],
            ["Sharpe Ratio", "Max Drawdown (%)", "Avg Turnover"]):
        bars = ax.bar(names, vals, color=colors, edgecolor="black", lw=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.001 * max(vals + [1e-9]),
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Test Set Metrics (2023–2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig_list.append(fig5)

    print(f"Generated {len(fig_list)} figures.")
    return fig_list


def save_pdf(fig_list, test_metrics,
             pdf_path="rl_portfolio_report_v3.pdf"):
    with pdf_backend.PdfPages(pdf_path) as pdf:
        # Cover page
        fig_c, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        lines = [
            (0.5, 0.78, "RL Portfolio Optimization v3",  20, "bold"),
            (0.5, 0.64, "Phase 1: DCC-GARCH GAT + HMM Regime + Expanded Universe", 14, "normal"),
            (0.5, 0.52, "Assets: " + " | ".join(ASSETS), 11, "normal"),
            (0.5, 0.42, f"Train: 2000–2019 | Val: 2020–2022 | Test: 2023–2024", 11, "normal"),
            (0.5, 0.32, f"Algorithm: PPO + CVaR (ALM)  |  N_ACTIONS={N_ACTIONS}", 11, "normal"),
        ]
        for x, y, txt, fs, fw in lines:
            ax.text(x, y, txt, ha="center", fontsize=fs,
                    fontweight=fw, transform=ax.transAxes)
        pdf.savefig(fig_c, bbox_inches="tight"); plt.close(fig_c)

        # Summary table
        fig_t, ax = plt.subplots(figsize=(12, 3))
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
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor("#2c3e50")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        ax.set_title("Test Results (2023–2024)",
                     fontsize=13, fontweight="bold", pad=20)
        pdf.savefig(fig_t, bbox_inches="tight"); plt.close(fig_t)

        for fig in fig_list:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        pdf.infodict()["Title"] = "RL Portfolio Optimization v3"

    print(f"PDF saved → {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── Data ─────────────────────────────────────────────────────────────────
    full_prices    = download_data(ASSETS, DATA_START, DATA_END)
    feature_engine = FeatureEngine(full_prices)

    def get_date_indices(start_str, end_str, min_history=252):
        dates = feature_engine.dates
        mask  = (dates >= start_str) & (dates <= end_str)
        idx   = np.where(mask)[0]
        return idx[idx >= min_history].tolist()

    train_idx = get_date_indices(TRAIN_START, TRAIN_END)
    val_idx   = get_date_indices(VAL_START,   VAL_END)
    test_idx  = get_date_indices(TEST_START,  TEST_END)
    print(f"\nTrain: {len(train_idx)} | Val: {len(val_idx)} "
          f"| Test: {len(test_idx)} days")

    # ── Adaptive ZETA ────────────────────────────────────────────────────────
    all_flat     = feature_engine._returns_np.flatten()
    cvar_natural = float(np.mean(all_flat[
        all_flat <= np.percentile(all_flat, 5)]))
    global ZETA
    ZETA = abs(cvar_natural) * 0.1
    print(f"Natural CVaR_5% = {cvar_natural:.4f}  →  ZETA = {ZETA:.4f}")

    # ── Training ─────────────────────────────────────────────────────────────
    results = {}
    results["PPO_CVaR"] = train_ppo("PPO_CVaR", feature_engine, train_idx)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Capture test-period regime posteriors for plot ────────────────────────
    test_posteriors = np.stack(
        [feature_engine.hmm.get_posterior(t) for t in test_idx])
    results["PPO_CVaR"]["test_posteriors"] = test_posteriors

    # ── Validation ───────────────────────────────────────────────────────────
    print("\nValidation set (2020–2022) ...")
    m = evaluate_policy(results["PPO_CVaR"]["actor"],
                        feature_engine, val_idx, n_episodes=3)
    print(f"  PPO_CVaR: CumRet={m['cum_return']:+.2f}% | "
          f"Sharpe={m['sharpe']:.2f} | MDD={m['max_drawdown']:.2f}% | "
          f"CVaR={m['cvar_95']:.4f}")

    # ── Test ─────────────────────────────────────────────────────────────────
    print("\nTest set (2023–2024) ...")
    test_metrics = {}
    m = evaluate_policy(results["PPO_CVaR"]["actor"],
                        feature_engine, test_idx,
                        episode_len=len(test_idx) - 1, n_episodes=1)
    test_metrics["PPO_CVaR"] = m

    baseline = compute_equal_weight_baseline(feature_engine, test_idx)
    test_metrics["EqualWeight"] = baseline

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("TEST SET RESULTS  (2023–2024)")
    print("="*75)
    print(f"{'Experiment':<20} {'Cum.Return':>12} {'Sharpe':>8} "
          f"{'MaxDrawdown':>13} {'CVaR_0.95':>11} {'Turnover':>10}")
    print("-"*75)
    for name, m in test_metrics.items():
        print(f"{name:<20} {m['cum_return']:>11.2f}% "
              f"{m['sharpe']:>8.3f} "
              f"{m['max_drawdown']:>12.2f}% "
              f"{m['cvar_95']:>11.4f} "
              f"{m['turnover']:>10.4f}")
    print("="*75)

    # ── Plots & PDF ───────────────────────────────────────────────────────────
    fig_list = generate_plots(results, test_metrics)
    save_pdf(fig_list, test_metrics)
    print("\nDone!")


if __name__ == "__main__":
    main()
