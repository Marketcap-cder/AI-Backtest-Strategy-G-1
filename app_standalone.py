"""
app_standalone.py -- Memecoin Backtester + Security Audit (Streamlit), single-file bundle

Launch:   pip install streamlit plotly pandas numpy aiohttp   (numba optional)
          streamlit run app_standalone.py
Headless: python app_standalone.py --selftest      (offline demo pipeline, no Streamlit UI)

This file is self-contained. The three library modules (memecoin_quant, dexscreener_adapter,
safety_module) are embedded below as source strings and registered in sys.modules at import
time, so the app code can use ordinary `import` statements and each module keeps its own
namespace (both dexscreener_adapter and safety_module define a SafetyDataProvider; they differ).
Generated from the separate files; edit the app section (below the embedded block) freely.

Simulation only. Not financial advice.
"""
from __future__ import annotations

import sys
import types

# =============================================================================
# EMBEDDED MODULES (verbatim copies of the standalone files)
# =============================================================================
_EMBEDDED = {
    'memecoin_quant': r'''"""
memecoin_quant.py -- Memecoin Backtester + On-Chain Coin Suggester (foundation)

Classes (all swappable via ABCs):
    DataIngestion        -> MarketDataSource (CSV / mock / your live API)
    DiscoveryEngine      -> DiscoverySource  (async polling) + SIR attention + SafetyChecker
    StrategyEngine       -> VectorStrategy (numba path) / EventStrategy (stateful path)
    AMMExecutionSim      -> constant-product (x*y=k) execution with gas/tax/fees/lots
    PerformanceAnalytics -> metrics from equity curves + trade PnL

Conventions
    * "quote" = native asset the pool is priced in (SOL/ETH/BNB); price = quote per token.
    * Vectorized engine: decisions use bar i-1 information, fills use bar i pool state
      (no lookahead). Our own trades do NOT persist in the pool between bars (documented
      simplification); the event engine can opt into pool mutation.
    * All chain constants are PLACEHOLDERS -- calibrate against real fills.
    * Regex contract scanning is a heuristic. Production needs bytecode analysis and
      live sell simulation (eth_call / simulateTransaction) feeding `sell_simulation_ok`.
"""
from __future__ import annotations

import abc
import asyncio
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain as ichain
from typing import (Any, Awaitable, Callable, Dict, Iterable, List, Mapping,
                    Optional, Sequence, Set, Tuple, Type)

import numpy as np
import pandas as pd

try:  # Numba is optional; the code runs (slower) without it.
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn: Callable) -> Callable:
            return fn
        return deco

log = logging.getLogger("memecoin_quant")


# =============================================================================
# 0. Core types & chain config
# =============================================================================
class Chain(str, Enum):
    ETHEREUM = "ethereum"
    BSC = "bsc"
    BASE = "base"
    SOLANA = "solana"


@dataclass(frozen=True)
class ChainConfig:
    chain: Chain
    quote_symbol: str
    base_gas_quote: float      # base cost per tx, in quote units (incl. typical tip)
    priority_scale: float      # gas = base * (1 + priority_scale * congestion)
    token_lot: float           # smallest tradable token increment (decimals / tick)
    min_trade_quote: float     # dust floor for a swap
    pool_fee_bps: float        # LP fee (Raydium ~25, Uniswap v2 30)


CHAINS: Dict[Chain, ChainConfig] = {  # placeholders -- calibrate!
    Chain.SOLANA: ChainConfig(Chain.SOLANA, "SOL", 0.0005, 10.0, 1e-6, 0.01, 25.0),
    Chain.ETHEREUM: ChainConfig(Chain.ETHEREUM, "ETH", 0.004, 4.0, 1e-9, 0.01, 30.0),
    Chain.BSC: ChainConfig(Chain.BSC, "BNB", 0.0006, 3.0, 1e-9, 0.01, 25.0),
    Chain.BASE: ChainConfig(Chain.BASE, "ETH", 0.00005, 5.0, 1e-9, 0.005, 30.0),
}


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class PoolState:
    reserve_token: float
    reserve_quote: float
    fee_bps: float = 25.0

    @property
    def mid_price(self) -> float:
        return self.reserve_quote / self.reserve_token

    @property
    def k(self) -> float:
        return self.reserve_token * self.reserve_quote

    def copy(self) -> "PoolState":
        return PoolState(self.reserve_token, self.reserve_quote, self.fee_bps)


# =============================================================================
# 1. AMM execution simulator  (Uniswap v2 constant product)
# =============================================================================
def cp_amount_out(reserve_out: float, reserve_in: float, amount_in: float,
                  fee_bps: float = 0.0) -> float:
    """dx = X*dy / (Y + dy), with X = reserve_out, Y = reserve_in, dy net of LP fee."""
    if amount_in <= 0.0 or reserve_in <= 0.0 or reserve_out <= 0.0:
        return 0.0
    eff = amount_in * (1.0 - fee_bps / 1e4)
    return reserve_out * eff / (reserve_in + eff)


@dataclass
class Fill:
    side: Side
    success: bool
    reason: str = ""
    tokens: float = 0.0          # BUY: tokens received (post-tax); SELL: tokens sold
    quote: float = 0.0           # BUY: quote sent to swap; SELL: gross quote out of pool
    net_quote: float = 0.0       # SELL: proceeds after marketing fee and gas
    gas: float = 0.0
    total_cost: float = 0.0      # BUY: quote + gas
    avg_price: float = 0.0
    mid_before: float = 0.0
    slippage_bps: float = 0.0    # total friction vs mid (curve + LP fee + tax + marketing)
    price_impact_bps: float = 0.0  # pure curve impact


class AMMExecutionSim:
    """Dynamic-slippage execution against a constant-product pool."""

    def __init__(self, cfg: ChainConfig, buy_tax_bps: float = 0.0,
                 sell_tax_bps: float = 0.0, marketing_fee_bps: float = 0.0,
                 max_price_impact_bps: float = 5000.0) -> None:
        self.cfg = cfg
        self.buy_tax = buy_tax_bps / 1e4
        self.sell_tax = sell_tax_bps / 1e4
        self.mkt = marketing_fee_bps / 1e4
        self.max_impact_bps = max_price_impact_bps

    def gas_cost(self, congestion: float = 0.0) -> float:
        c = min(max(congestion, 0.0), 1.0)
        return self.cfg.base_gas_quote * (1.0 + self.cfg.priority_scale * c)

    def _quantize(self, tokens: float) -> float:
        lot = self.cfg.token_lot
        return math.floor(tokens / lot) * lot if lot > 0 else tokens

    def buy(self, pool: PoolState, quote_in: float, congestion: float = 0.0,
            mutate: bool = False) -> Fill:
        gas = self.gas_cost(congestion)
        if quote_in < self.cfg.min_trade_quote:
            return Fill(Side.BUY, False, "below min trade")
        swap_in = quote_in * (1.0 - self.mkt)
        eff = swap_in * (1.0 - pool.fee_bps / 1e4)
        impact = eff / (pool.reserve_quote + eff) * 1e4
        if impact > self.max_impact_bps:
            return Fill(Side.BUY, False, "price impact too high", price_impact_bps=impact)
        gross = cp_amount_out(pool.reserve_token, pool.reserve_quote, swap_in, pool.fee_bps)
        tokens = self._quantize(gross * (1.0 - self.buy_tax))
        if tokens <= 0:
            return Fill(Side.BUY, False, "zero output")
        mid = pool.mid_price
        ideal = quote_in / mid
        fill = Fill(Side.BUY, True, tokens=tokens, quote=quote_in, gas=gas,
                    total_cost=quote_in + gas, avg_price=quote_in / tokens, mid_before=mid,
                    slippage_bps=(1.0 - tokens / ideal) * 1e4, price_impact_bps=impact)
        if mutate:
            pool.reserve_quote += swap_in
            pool.reserve_token -= gross
        return fill

    def sell(self, pool: PoolState, tokens_in: float, congestion: float = 0.0,
             mutate: bool = False) -> Fill:
        gas = self.gas_cost(congestion)
        tokens = self._quantize(tokens_in)
        if tokens <= 0:
            return Fill(Side.SELL, False, "zero size")
        to_pool = tokens * (1.0 - self.sell_tax)
        eff = to_pool * (1.0 - pool.fee_bps / 1e4)
        impact = eff / (pool.reserve_token + eff) * 1e4
        if impact > self.max_impact_bps:
            return Fill(Side.SELL, False, "price impact too high", price_impact_bps=impact)
        gross = cp_amount_out(pool.reserve_quote, pool.reserve_token, to_pool, pool.fee_bps)
        net = gross * (1.0 - self.mkt) - gas
        if net <= 0:
            return Fill(Side.SELL, False, "proceeds below gas", price_impact_bps=impact)
        mid = pool.mid_price
        ideal = tokens * mid
        fill = Fill(Side.SELL, True, tokens=tokens, quote=gross, net_quote=net, gas=gas,
                    avg_price=net / tokens, mid_before=mid,
                    slippage_bps=(1.0 - net / ideal) * 1e4, price_impact_bps=impact)
        if mutate:
            pool.reserve_token += to_pool
            pool.reserve_quote -= gross
        return fill

    def liquidation_value(self, pool: PoolState, tokens: float) -> float:
        """Mark-to-market through the curve (honest mark for illiquid bags), no gas."""
        if tokens <= 0:
            return 0.0
        gross = cp_amount_out(pool.reserve_quote, pool.reserve_token,
                              tokens * (1.0 - self.sell_tax), pool.fee_bps)
        return gross * (1.0 - self.mkt)


# =============================================================================
# 2. Spike & decay: SIR attention model, half-life decay, late-buyer penalty
# =============================================================================
@dataclass(frozen=True)
class SIRParams:
    beta: float        # transmission rate (1/hour)
    gamma: float       # recovery rate (1/hour) -> post-peak dump speed
    population: float  # reachable audience N

    @property
    def r0(self) -> float:
        return self.beta / self.gamma if self.gamma > 0 else float("inf")

    @property
    def half_life_hours(self) -> float:
        return math.log(2.0) / self.gamma if self.gamma > 0 else float("inf")


def simulate_sir(p: SIRParams, i0: float, steps: int, dt_hours: float,
                 substeps: int = 10) -> pd.DataFrame:
    """Euler-integrated SIR: dS=-bSI/N, dI=bSI/N-gI, dR=gI."""
    s, i, r = p.population - i0, i0, 0.0
    h = dt_hours / substeps
    out = np.empty((steps, 3))
    for t in range(steps):
        for _ in range(substeps):
            new_inf = p.beta * s * i / p.population * h
            new_rec = p.gamma * i * h
            s -= new_inf
            i += new_inf - new_rec
            r += new_rec
        out[t] = (s, i, r)
    return pd.DataFrame(out, columns=["S", "I", "R"])


def estimate_sir_params(mentions: np.ndarray, dt_hours: float = 1.0,
                        default_gamma: float = 0.5) -> SIRParams:
    """
    Heuristic fit: log-linear growth rate r on the ascent, decay rate d after the peak.
    gamma ~ d, beta = r + gamma (early phase, S ~ N). N from I_peak/N = 1 - (1+ln R0)/R0.
    """
    x = pd.Series(np.asarray(mentions, float)).rolling(3, min_periods=1).mean().to_numpy()
    n = x.size
    if n < 6 or x.max() <= 0:
        return SIRParams(default_gamma * 1.5, default_gamma, max(float(x.sum()), 1.0) * 5)
    p = int(np.argmax(x))
    peak = x[p]

    def slope(idx: np.ndarray) -> Optional[float]:
        if idx.size < 3:
            return None
        return float(np.polyfit(idx * dt_hours, np.log(x[idx]), 1)[0])

    r = slope(np.arange(p + 1)[x[:p + 1] >= 0.1 * peak])
    d = slope(np.arange(p, n)[x[p:] >= 0.05 * peak])
    gamma = -d if (d is not None and d < 0) else default_gamma
    gamma = max(gamma, 1e-3)
    beta = max(r if r is not None else 0.0, 0.0) + gamma
    r0 = beta / gamma
    frac = 1.0 - (1.0 + math.log(r0)) / r0 if r0 > 1.0 else 0.1
    return SIRParams(beta, gamma, peak / max(frac, 1e-3))


def attention_decay(attention0: float, t_hours: float, half_life_hours: float) -> float:
    """Attention(t) = Attention0 * exp(-lambda t), lambda = ln2 / half-life."""
    lam = math.log(2.0) / half_life_hours
    return attention0 * math.exp(-lam * t_hours)


@dataclass(frozen=True)
class LateBuyerPenalty:
    grace_hours: float = 0.5       # no penalty this long after the velocity peak
    half_life_hours: float = 2.0   # score halves every this many hours beyond grace
    floor: float = 0.05

    def multiplier(self, hours_since_peak: float) -> float:
        if hours_since_peak <= self.grace_hours:
            return 1.0
        k = math.log(2.0) / self.half_life_hours
        return self.floor + (1.0 - self.floor) * math.exp(-k * (hours_since_peak - self.grace_hours))


# =============================================================================
# 3. Rug-pull / honeypot safety checker
# =============================================================================
ZERO_ADDRESSES: Set[str] = {
    "0x" + "0" * 40,
    "0x000000000000000000000000000000000000dead",
    "11111111111111111111111111111111",  # Solana system program (null authority)
}
NON_CIRCULATING_LABELS = {"exchange", "lp", "burn", "locker"}


@dataclass
class HolderRecord:
    address: str
    balance: float
    label: str = "wallet"            # wallet | exchange | lp | burn | locker | contract
    funder: Optional[str] = None     # address that first funded this wallet (cluster key)


@dataclass
class ContractSnapshot:
    address: str
    chain: Chain
    owner: Optional[str] = None            # EVM owner(); None => no owner() / renounced
    source_code: str = ""
    buy_tax_bps: float = 0.0
    sell_tax_bps: float = 0.0
    lp_locked_pct: float = 0.0
    lp_burned_pct: float = 0.0
    mint_authority: Optional[str] = None   # Solana SPL
    freeze_authority: Optional[str] = None # Solana SPL
    total_supply: float = 0.0
    holders: List[HolderRecord] = field(default_factory=list)
    sell_simulation_ok: Optional[bool] = None  # live simulation result if you have it


@dataclass(frozen=True)
class Flag:
    code: str
    weight: float
    message: str
    critical: bool = False


@dataclass
class SafetyReport:
    passed: bool
    risk_score: float
    flags: List[Flag]
    metrics: Dict[str, float]


@dataclass(frozen=True)
class SourcePattern:
    code: str
    regex: "re.Pattern[str]"
    weight: float
    critical: bool
    message: str


_I = re.I
SOURCE_PATTERNS: Tuple[SourcePattern, ...] = (
    SourcePattern("BLACKLIST", re.compile(r"(blacklist|blocklist|isBot\b|_bots?\b|antiBot)", _I), 35, False,
                  "Blacklist/bot-list logic can block individual sellers"),
    SourcePattern("TRADE_FREEZE", re.compile(
        r"(tradingEnabled|tradingOpen|tradingActive|whenNotPaused|\bpause\s*\(|setTrading\w*|disableTrading)", _I),
        35, False, "Owner-toggleable trading switch / pause can freeze trades"),
    SourcePattern("TAX_MUTABLE", re.compile(
        r"(function\s+(set|update|change)\w*(tax|fee)\w*|_?(buy|sell)(tax|fee)\s*=)", _I), 30, False,
        "Tax/fee can be modified after launch"),
    SourcePattern("LIMITS_MUTABLE", re.compile(
        r"function\s+(set|update|remove)\w*(maxtx|maxwallet|maxtransaction|limits?)", _I), 15, False,
        "Max-tx / max-wallet limits are mutable"),
    SourcePattern("MINT", re.compile(r"function\s+_?mint\w*", _I), 40, True,
                  "Mint function present (unbounded supply risk)"),
    SourcePattern("SELFDESTRUCT", re.compile(r"selfdestruct\s*\("), 40, True, "selfdestruct present"),
    SourcePattern("DELEGATECALL", re.compile(r"delegatecall\s*\("), 25, False,
                  "delegatecall / upgradeable logic can change behavior"),
)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _extract_functions(src: str) -> List[Tuple[str, str, str]]:
    """(name, header, body) via brace matching; skips bodyless declarations."""
    out: List[Tuple[str, str, str]] = []
    for m in re.finditer(r"function\s+(\w+)\s*\(", src):
        brace = src.find("{", m.end())
        semi = src.find(";", m.end())
        if brace == -1 or (semi != -1 and semi < brace):
            continue
        depth, j = 0, brace
        while j < len(src):
            c = src[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1), src[m.start():brace], src[brace:j + 1]))
    return out


class HolderAnalyzer:
    @staticmethod
    def gini(balances: Sequence[float]) -> float:
        x = np.sort(np.asarray([b for b in balances if b > 0], float))
        n = x.size
        if n == 0 or x.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float(2.0 * (idx * x).sum() / (n * x.sum()) - (n + 1.0) / n)

    @classmethod
    def analyze(cls, holders: Sequence[HolderRecord], total_supply: float,
                top_n: int = 10, min_cluster: int = 3) -> Dict[str, float]:
        excluded = sum(h.balance for h in holders if h.label in NON_CIRCULATING_LABELS)
        circ = max(total_supply - excluded, 1e-12) if total_supply > 0 else max(
            sum(h.balance for h in holders if h.label not in NON_CIRCULATING_LABELS), 1e-12)
        real = sorted((h for h in holders if h.label not in NON_CIRCULATING_LABELS),
                      key=lambda h: h.balance, reverse=True)
        top_share = sum(h.balance for h in real[:top_n]) / circ
        clusters: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for h in real:
            if h.funder:
                clusters[h.funder] += h.balance
                counts[h.funder] += 1
        big = [v for k, v in clusters.items() if counts[k] >= min_cluster]
        return {
            "top_n_share": float(top_share),
            "gini": cls.gini([h.balance for h in real]),
            "largest_cluster_share": float(max(big) / circ) if big else 0.0,
            "n_holders": float(len(real)),
        }


class SafetyChecker:
    def __init__(self, max_risk: float = 50.0, probe_quote: float = 0.1,
                 max_roundtrip_loss: float = 0.30) -> None:
        self.max_risk = max_risk
        self.probe_quote = probe_quote
        self.max_roundtrip_loss = max_roundtrip_loss

    @staticmethod
    def owner_renounced(snap: ContractSnapshot) -> bool:
        return snap.owner is None or snap.owner.lower() in {a.lower() for a in ZERO_ADDRESSES}

    def check(self, snap: ContractSnapshot, pool: Optional[PoolState] = None) -> SafetyReport:
        flags: List[Flag] = []
        metrics: Dict[str, float] = {}
        renounced = self.owner_renounced(snap)

        # --- ownership / authorities
        if snap.chain is Chain.SOLANA:
            if snap.freeze_authority:
                flags.append(Flag("FREEZE_AUTHORITY", 50, "Freeze authority active", True))
            if snap.mint_authority:
                flags.append(Flag("MINT_AUTHORITY", 40, "Mint authority active", True))
        elif not renounced:
            flags.append(Flag("OWNER_NOT_RENOUNCED", 15, f"owner != 0x000... ({snap.owner})"))

        # --- source parsing
        if snap.source_code:
            src = _COMMENT_RE.sub("", snap.source_code)
            fns = _extract_functions(src)
            for pat in SOURCE_PATTERNS:
                hits = [(n, h) for n, h, b in fns if pat.regex.search(h + b)]
                if hits:
                    inert = renounced and all("onlyowner" in h.lower() for _, h in hits)
                    scale = 0.2 if inert else 1.0
                    flags.append(Flag(pat.code, pat.weight * scale,
                                      pat.message + (" (owner-gated, owner renounced)" if inert else
                                                     f" in {[n for n, _ in hits][:3]}"),
                                      pat.critical and not inert))
                elif pat.regex.search(src):
                    flags.append(Flag(pat.code, pat.weight * 0.4, pat.message + " (declared, no setter found)"))
        else:
            flags.append(Flag("UNVERIFIED_SOURCE", 25, "No verified source supplied"))

        # --- taxes
        if snap.sell_tax_bps >= 5000:
            flags.append(Flag("HONEYPOT_TAX", 100, f"sell tax {snap.sell_tax_bps/100:.0f}%", True))
        elif snap.sell_tax_bps >= 2500:
            flags.append(Flag("EXCESSIVE_SELL_TAX", 40, f"sell tax {snap.sell_tax_bps/100:.0f}%"))
        if snap.buy_tax_bps >= 2500:
            flags.append(Flag("EXCESSIVE_BUY_TAX", 25, f"buy tax {snap.buy_tax_bps/100:.0f}%"))

        # --- liquidity lock
        lp_safe = snap.lp_locked_pct + snap.lp_burned_pct
        metrics["lp_safe_pct"] = lp_safe
        if lp_safe < 80:
            flags.append(Flag("LP_UNLOCKED", 40 if lp_safe < 20 else 20,
                              f"only {lp_safe:.0f}% of LP locked/burned"))

        # --- sell simulation
        if snap.sell_simulation_ok is False:
            flags.append(Flag("HONEYPOT_SELL_REVERT", 100, "Live sell simulation reverted", True))
        if pool is not None:
            sim = AMMExecutionSim(CHAINS[snap.chain], snap.buy_tax_bps, snap.sell_tax_bps)
            b = sim.buy(pool, self.probe_quote)
            if b.success:
                s = sim.sell(pool, b.tokens)
                loss = 1.0 - (s.net_quote / b.total_cost) if s.success else 1.0
                metrics["roundtrip_loss"] = loss
                if loss > self.max_roundtrip_loss:
                    flags.append(Flag("ROUNDTRIP_LOSS", 40, f"probe round-trip loses {loss:.0%}",
                                      loss > 0.7))

        # --- holders
        if snap.holders:
            hm = HolderAnalyzer.analyze(snap.holders, snap.total_supply)
            metrics.update(hm)
            if hm["top_n_share"] > 0.5:
                flags.append(Flag("TOP_HEAVY", 40, f"top10 non-exchange hold {hm['top_n_share']:.0%}"))
            elif hm["top_n_share"] > 0.3:
                flags.append(Flag("TOP_HEAVY_MILD", 20, f"top10 non-exchange hold {hm['top_n_share']:.0%}"))
            if hm["gini"] > 0.9:
                flags.append(Flag("HIGH_GINI", 15, f"gini {hm['gini']:.2f}"))
            if hm["largest_cluster_share"] > 0.2:
                flags.append(Flag("INSIDER_CLUSTER", 30,
                                  f"co-funded wallet cluster holds {hm['largest_cluster_share']:.0%}"))

        risk = min(100.0, sum(f.weight for f in flags))
        passed = risk < self.max_risk and not any(f.critical for f in flags)
        return SafetyReport(passed, risk, flags, metrics)


# =============================================================================
# 4. Data ingestion
# =============================================================================
REQUIRED_COLS = ("close", "volume", "reserve_quote")
OPTIONAL_DEFAULTS = {"social_mentions": 0.0, "congestion": 0.0}


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    token: str
    close: float
    volume: float
    reserve_quote: float
    reserve_token: float
    social_mentions: float = 0.0
    congestion: float = 0.0


@dataclass(frozen=True)
class WalletEvent:
    ts: pd.Timestamp
    wallet: str
    token: str
    side: Side
    size_quote: float


@dataclass
class FeedItem:
    ts: pd.Timestamp
    bars: Dict[str, Bar]
    wallet_events: List[WalletEvent] = field(default_factory=list)


class MarketDataSource(abc.ABC):
    @abc.abstractmethod
    def load(self, token: str) -> pd.DataFrame:
        """Return DatetimeIndex frame with REQUIRED_COLS (+ optional social_mentions, congestion)."""


class CsvDataSource(MarketDataSource):
    def __init__(self, path_template: str = "data/{token}.csv", ts_col: str = "timestamp") -> None:
        self.path_template, self.ts_col = path_template, ts_col

    def load(self, token: str) -> pd.DataFrame:
        df = pd.read_csv(self.path_template.format(token=token), parse_dates=[self.ts_col])
        return df.set_index(self.ts_col).sort_index()


def generate_mock_market(n_bars: int = 2880, seed: int = 7, start: str = "2026-01-01",
                         freq: str = "1min", rug_at: Optional[int] = None,
                         beta: float = 2.0, gamma: float = 0.5, delay_bars: int = 120,
                         init_liq_quote: float = 40.0, p0: float = 1e-6) -> pd.DataFrame:
    """Spike & decay market: SIR attention drives price + liquidity; optional rug."""
    rng = np.random.default_rng(seed)
    dt_h = pd.Timedelta(freq).total_seconds() / 3600.0
    sir = simulate_sir(SIRParams(beta, gamma, 1e5), i0=20, steps=n_bars - delay_bars, dt_hours=dt_h)
    infected = np.concatenate([np.full(delay_bars, 20.0), sir["I"].to_numpy()])
    mentions = rng.poisson(np.maximum(infected * 0.05, 0.1)).astype(float)
    att = pd.Series(mentions).rolling(10, min_periods=1).mean().to_numpy()
    att_n = att / max(att.max(), 1e-9)
    noise = np.cumsum(rng.normal(0, 0.004, n_bars))
    close = p0 * np.exp(2.5 * np.log1p(att / 5.0) + noise)
    rq = init_liq_quote * (1.0 + 4.0 * att_n) * np.exp(rng.normal(0, 0.003, n_bars))
    if rug_at is not None and 0 < rug_at < n_bars:
        close[rug_at:] *= 0.03
        rq[rug_at:] *= 0.03
    ret = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
    vol = rq * (ret * 20 + rng.random(n_bars) * 0.02)
    idx = pd.date_range(start, periods=n_bars, freq=freq)
    return pd.DataFrame({"close": close, "volume": vol, "reserve_quote": rq,
                         "social_mentions": mentions, "congestion": att_n}, index=idx)


class MockDataSource(MarketDataSource):
    def __init__(self, specs: Optional[Mapping[str, Dict[str, Any]]] = None) -> None:
        self.specs = dict(specs or {})

    def load(self, token: str) -> pd.DataFrame:
        return generate_mock_market(**self.specs.get(token, {}))


class DataIngestion:
    def __init__(self, source: MarketDataSource, max_concurrency: int = 8) -> None:
        self.source, self.max_concurrency = source, max_concurrency

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("index must be DatetimeIndex")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        for c, v in OPTIONAL_DEFAULTS.items():
            if c not in df.columns:
                df[c] = v
        df = df[(df["close"] > 0) & (df["reserve_quote"] > 0)]
        return df.astype(float)

    def load(self, token: str) -> pd.DataFrame:
        return self.validate(self.source.load(token))

    async def load_many(self, tokens: Sequence[str]) -> Dict[str, pd.DataFrame]:
        sem = asyncio.Semaphore(self.max_concurrency)
        loop = asyncio.get_running_loop()

        async def one(tok: str) -> Tuple[str, pd.DataFrame]:
            async with sem:  # swap run_in_executor for native async IO in your live source
                return tok, await loop.run_in_executor(None, self.load, tok)
        return dict(await asyncio.gather(*(one(t) for t in tokens)))

    @staticmethod
    def to_event_feed(frames: Mapping[str, pd.DataFrame],
                      wallet_events: Sequence[WalletEvent] = ()) -> List[FeedItem]:
        """Wallet events must be snapped to bar timestamps."""
        parts = []
        for tok, df in frames.items():
            d = df.copy()
            d["token"] = tok
            parts.append(d)
        big = pd.concat(parts).sort_index()
        wl: Dict[pd.Timestamp, List[WalletEvent]] = defaultdict(list)
        for ev in wallet_events:
            wl[ev.ts].append(ev)
        feed: List[FeedItem] = []
        for ts, g in big.groupby(level=0, sort=True):
            bars = {r.token: Bar(ts, r.token, r.close, r.volume, r.reserve_quote,
                                 r.reserve_quote / r.close, r.social_mentions, r.congestion)
                    for r in g.itertuples()}
            feed.append(FeedItem(ts, bars, wl.get(ts, [])))
        return feed


# =============================================================================
# 5. Performance analytics
# =============================================================================
class PerformanceAnalytics:
    @staticmethod
    def summarize(equity: Sequence[float], trade_pnls: Optional[Sequence[float]] = None,
                  bars_per_year: float = 525_600.0) -> Dict[str, float]:
        eq = np.asarray(equity, float)
        if eq.size < 2:
            return {}
        prev = eq[:-1]
        rets = np.diff(eq) / np.where(prev > 0, prev, np.nan)
        rets = rets[np.isfinite(rets)]
        std = rets.std() if rets.size else 0.0
        down = rets[rets < 0]
        dstd = down.std() if down.size > 1 else 0.0
        dd = eq / np.maximum.accumulate(eq) - 1.0
        total = eq[-1] / eq[0] - 1.0
        out = {
            "total_return": float(total),
            "max_drawdown": float(dd.min()),
            "sharpe": float(rets.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0,
            "sortino": float(rets.mean() / dstd * math.sqrt(bars_per_year)) if dstd > 0 else 0.0,
            "calmar": float(total / abs(dd.min())) if dd.min() < 0 else 0.0,
        }
        if trade_pnls is not None and len(trade_pnls):
            p = np.asarray(trade_pnls, float)
            gains, losses = p[p > 0].sum(), -p[p < 0].sum()
            out.update({"n_trades": float(p.size), "win_rate": float((p > 0).mean()),
                        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
                        "expectancy": float(p.mean())})
        else:
            out["n_trades"] = 0.0
        return out


# =============================================================================
# 6. Vectorized engine (NumPy + Numba) and strategy ABCs
# =============================================================================
@njit(cache=False)
def ema_nb(x: np.ndarray, span: int) -> np.ndarray:
    out = np.empty_like(x)
    a = 2.0 / (span + 1.0)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


@njit(cache=False)
def _cp_out(r_out: float, r_in: float, amt: float, fee: float) -> float:
    if amt <= 0.0:
        return 0.0
    eff = amt * (1.0 - fee)
    return r_out * eff / (r_in + eff)


@njit(cache=False)
def vector_backtest_kernel(close, rq, gas, entries, exits, size_quote, pool_fee, buy_tax,
                           sell_tax, mkt_fee, stop_loss, take_profit, trail, init_cash):
    """Long-only, one position. Decide on bar i-1, fill on bar i via x*y=k.
    Equity is marked THROUGH the curve (liquidation value)."""
    n = close.shape[0]
    equity = np.empty(n)
    pnls = np.empty(n)
    n_tr = 0
    cash = init_cash
    tokens = 0.0
    in_pos = False
    entry_cost = 0.0
    entry_ref = 1.0
    peak = 0.0
    for i in range(n):
        q = rq[i]
        p = close[i]
        t = q / p
        if i > 0:
            pp = close[i - 1]
            if in_pos:
                if pp > peak:
                    peak = pp
                r = pp / entry_ref - 1.0
                want = exits[i - 1] or r <= -stop_loss or r >= take_profit
                if (not want) and trail > 0.0 and pp <= peak * (1.0 - trail):
                    want = True
                if want:
                    out = _cp_out(q, t, tokens * (1.0 - sell_tax), pool_fee) * (1.0 - mkt_fee) - gas[i]
                    if out < 0.0:
                        out = 0.0
                    cash += out
                    pnls[n_tr] = out - entry_cost
                    n_tr += 1
                    tokens = 0.0
                    in_pos = False
            elif entries[i - 1] and cash >= size_quote + gas[i]:
                gross = _cp_out(t, q, size_quote * (1.0 - mkt_fee), pool_fee)
                tk = gross * (1.0 - buy_tax)
                if tk > 0.0:
                    cash -= size_quote + gas[i]
                    tokens = tk
                    entry_cost = size_quote + gas[i]
                    entry_ref = p
                    peak = p
                    in_pos = True
        mtm = 0.0
        if in_pos:
            mtm = _cp_out(q, t, tokens * (1.0 - sell_tax), pool_fee) * (1.0 - mkt_fee)
        equity[i] = cash + mtm
    return equity, pnls[:n_tr], n_tr


class VectorStrategy(abc.ABC):
    """Inject custom rules here: return boolean entry/exit arrays aligned to df."""

    @abc.abstractmethod
    def generate(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        ...


class EmaMomentumStrategy(VectorStrategy):
    def __init__(self, fast: int = 5, slow: int = 20, min_mentions_growth: float = 0.0) -> None:
        self.fast, self.slow, self.g = int(fast), int(slow), min_mentions_growth

    def generate(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        c = df["close"].to_numpy(float)
        f, s = ema_nb(c, self.fast), ema_nb(c, self.slow)
        cross_up = (f > s) & (np.roll(f, 1) <= np.roll(s, 1))
        cross_dn = (f < s) & (np.roll(f, 1) >= np.roll(s, 1))
        m = df["social_mentions"].to_numpy(float)
        growth = m - np.roll(m, self.slow)
        entries = cross_up & (growth >= self.g)
        entries[0] = cross_dn[0] = False
        return entries, cross_dn


@dataclass
class VectorResult:
    equity: pd.Series
    pnls: np.ndarray
    metrics: Dict[str, float]


class VectorizedBacktester:
    def __init__(self, cfg: ChainConfig, buy_tax_bps: float = 0.0, sell_tax_bps: float = 0.0,
                 marketing_bps: float = 0.0, size_quote: float = 1.0, init_cash: float = 10.0,
                 stop_loss: float = np.inf, take_profit: float = np.inf, trail: float = 0.0,
                 bars_per_year: float = 525_600.0) -> None:
        self.cfg, self.size, self.cash = cfg, size_quote, init_cash
        self.bt, self.st, self.mk = buy_tax_bps / 1e4, sell_tax_bps / 1e4, marketing_bps / 1e4
        self.sl, self.tp, self.trail, self.bpy = stop_loss, take_profit, trail, bars_per_year

    def run(self, df: pd.DataFrame, strat: VectorStrategy) -> VectorResult:
        entries, exits = strat.generate(df)
        gas = self.cfg.base_gas_quote * (1.0 + self.cfg.priority_scale * np.clip(
            df["congestion"].to_numpy(float), 0.0, 1.0))
        eq, pnls, _ = vector_backtest_kernel(
            df["close"].to_numpy(float), df["reserve_quote"].to_numpy(float), gas,
            entries.astype(np.bool_), exits.astype(np.bool_), self.size,
            self.cfg.pool_fee_bps / 1e4, self.bt, self.st, self.mk,
            float(self.sl), float(self.tp), float(self.trail), self.cash)
        return VectorResult(pd.Series(eq, index=df.index),
                            pnls, PerformanceAnalytics.summarize(eq, pnls, self.bpy))

    def sweep(self, df: pd.DataFrame, strat_cls: Type[VectorStrategy],
              grid: Mapping[str, Sequence[Any]], sort_by: str = "sharpe") -> pd.DataFrame:
        names = list(grid)
        rows = []
        for combo in np.array(np.meshgrid(*[list(grid[n]) for n in names], indexing="ij")
                              ).reshape(len(names), -1).T:
            params = {n: (int(v) if float(v).is_integer() else float(v)) for n, v in zip(names, combo)}
            try:
                res = self.run(df, strat_cls(**params))
            except Exception as exc:  # noqa: BLE001
                log.warning("combo %s failed: %s", params, exc)
                continue
            rows.append({**params, **res.metrics})
        return pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)


# =============================================================================
# 7. Event-driven engine (state-dependent paths)
# =============================================================================
@dataclass
class Order:
    token: str
    side: Side
    quote_amount: float = 0.0   # BUY size
    fraction: float = 1.0       # SELL fraction of position
    tag: str = ""


@dataclass
class Position:
    tokens: float
    entry_cost: float
    entry_ts: pd.Timestamp
    entry_ref: float
    peak: float
    tag: str = ""


@dataclass
class TradeRecord:
    token: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    pnl: float
    pnl_pct: float
    exit_slippage_bps: float
    tag: str


class Portfolio:
    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.last_bar: Dict[str, Bar] = {}

    def mark(self, sim: AMMExecutionSim) -> float:
        v = self.cash
        for tok, pos in self.positions.items():
            b = self.last_bar[tok]
            v += sim.liquidation_value(PoolState(b.reserve_token, b.reserve_quote, sim.cfg.pool_fee_bps),
                                       pos.tokens)
        return v


@dataclass
class Context:
    ts: pd.Timestamp
    bars: Dict[str, Bar]
    portfolio: Portfolio
    closes: Dict[str, List[float]]
    mentions: Dict[str, List[float]]
    wallet_events: List[WalletEvent]


class EventStrategy(abc.ABC):
    @abc.abstractmethod
    def on_bar(self, ctx: Context) -> List[Order]:
        """Orders returned here fill at the NEXT timestamp's pool state."""


class TrailingStopMomentum(EventStrategy):
    def __init__(self, lookback: int = 30, trail_pct: float = 0.25, hard_stop_pct: float = 0.35,
                 size_quote: float = 1.0, max_positions: int = 2,
                 allow: Optional[Callable[[str], bool]] = None) -> None:
        self.lb, self.trail, self.hs = lookback, trail_pct, hard_stop_pct
        self.size, self.maxpos, self.allow = size_quote, max_positions, allow

    def on_bar(self, ctx: Context) -> List[Order]:
        orders: List[Order] = []
        for tok, pos in ctx.portfolio.positions.items():
            b = ctx.bars.get(tok)
            if b and (b.close <= pos.peak * (1 - self.trail) or b.close <= pos.entry_ref * (1 - self.hs)):
                orders.append(Order(tok, Side.SELL, tag="trail/stop"))
        slots = self.maxpos - len(ctx.portfolio.positions)
        for tok, b in ctx.bars.items():
            h, m = ctx.closes[tok], ctx.mentions[tok]
            if slots <= 0 or tok in ctx.portfolio.positions or len(h) <= self.lb:
                continue
            if self.allow and not self.allow(tok):
                continue
            if b.close > max(h[-self.lb - 1:-1]) and m[-1] > m[-self.lb]:
                orders.append(Order(tok, Side.BUY, quote_amount=self.size, tag="breakout"))
                slots -= 1
        return orders


class FollowWalletStrategy(EventStrategy):
    """Mirror tracked wallets; exit when they sell or a hard stop hits."""

    def __init__(self, tracked: Set[str], size_quote: float = 0.5, hard_stop_pct: float = 0.4,
                 allow: Optional[Callable[[str], bool]] = None) -> None:
        self.tracked, self.size, self.hs, self.allow = tracked, size_quote, hard_stop_pct, allow

    def on_bar(self, ctx: Context) -> List[Order]:
        orders: List[Order] = []
        for ev in ctx.wallet_events:
            if ev.wallet not in self.tracked:
                continue
            if ev.side is Side.BUY and ev.token not in ctx.portfolio.positions:
                if self.allow is None or self.allow(ev.token):
                    orders.append(Order(ev.token, Side.BUY, quote_amount=self.size, tag=f"follow:{ev.wallet}"))
            elif ev.side is Side.SELL and ev.token in ctx.portfolio.positions:
                orders.append(Order(ev.token, Side.SELL, tag="wallet-exit"))
        for tok, pos in ctx.portfolio.positions.items():
            b = ctx.bars.get(tok)
            if b and b.close <= pos.entry_ref * (1 - self.hs):
                orders.append(Order(tok, Side.SELL, tag="hard-stop"))
        return orders


@dataclass
class EventResult:
    equity: pd.Series
    trades: List[TradeRecord]
    metrics: Dict[str, float]


class EventDrivenBacktester:
    def __init__(self, sim: AMMExecutionSim, init_cash: float = 10.0,
                 bars_per_year: float = 525_600.0) -> None:
        self.sim, self.init_cash, self.bpy = sim, init_cash, bars_per_year

    def _execute(self, pf: Portfolio, o: Order, ts: pd.Timestamp, bars: Dict[str, Bar]) -> None:
        bar = bars.get(o.token)
        if bar is None:
            return
        pool = PoolState(bar.reserve_token, bar.reserve_quote, self.sim.cfg.pool_fee_bps)
        if o.side is Side.BUY:
            if o.token in pf.positions:
                return
            f = self.sim.buy(pool, o.quote_amount, bar.congestion)
            if not f.success or pf.cash < f.total_cost:
                return
            pf.cash -= f.total_cost
            pf.positions[o.token] = Position(f.tokens, f.total_cost, ts, bar.close, bar.close, o.tag)
            return
        pos = pf.positions.get(o.token)
        if pos is None:
            return
        frac = min(max(o.fraction, 0.0), 1.0)
        f = self.sim.sell(pool, pos.tokens * frac, bar.congestion)
        if not f.success:
            return  # can't exit (rugged/illiquid): position stays, marked through the curve
        pf.cash += f.net_quote
        cost = pos.entry_cost * (f.tokens / pos.tokens)
        pf.trades.append(TradeRecord(o.token, pos.entry_ts, ts, f.net_quote - cost,
                                     f.net_quote / cost - 1.0, f.slippage_bps, pos.tag or o.tag))
        pos.tokens -= f.tokens
        pos.entry_cost -= cost
        if frac >= 0.999999 or pos.tokens <= 0:
            del pf.positions[o.token]

    def run(self, feed: Sequence[FeedItem], strategy: EventStrategy) -> EventResult:
        pf = Portfolio(self.init_cash)
        closes: Dict[str, List[float]] = defaultdict(list)
        mentions: Dict[str, List[float]] = defaultdict(list)
        pending: List[Order] = []
        idx: List[pd.Timestamp] = []
        eq: List[float] = []
        for item in feed:
            for o in pending:
                self._execute(pf, o, item.ts, item.bars)
            pending = []
            for tok, b in item.bars.items():
                pf.last_bar[tok] = b
                closes[tok].append(b.close)
                mentions[tok].append(b.social_mentions)
                if tok in pf.positions:
                    pf.positions[tok].peak = max(pf.positions[tok].peak, b.close)
            pending = strategy.on_bar(Context(item.ts, item.bars, pf, closes, mentions, item.wallet_events))
            idx.append(item.ts)
            eq.append(pf.mark(self.sim))
        pnls = [t.pnl for t in pf.trades]
        return EventResult(pd.Series(eq, index=idx), pf.trades,
                           PerformanceAnalytics.summarize(eq, pnls, self.bpy))


class StrategyEngine:
    """Registry/factory so strategies can be swapped by name or config."""

    def __init__(self) -> None:
        self._vec: Dict[str, Type[VectorStrategy]] = {"ema_momentum": EmaMomentumStrategy}
        self._evt: Dict[str, Type[EventStrategy]] = {
            "trailing_momentum": TrailingStopMomentum, "follow_wallet": FollowWalletStrategy}

    def register_vector(self, name: str, cls: Type[VectorStrategy]) -> None:
        self._vec[name] = cls

    def register_event(self, name: str, cls: Type[EventStrategy]) -> None:
        self._evt[name] = cls

    def vector(self, name: str, **params: Any) -> VectorStrategy:
        return self._vec[name](**params)

    def event(self, name: str, **params: Any) -> EventStrategy:
        return self._evt[name](**params)


# =============================================================================
# 8. Discovery engine (async polling + scoring)
# =============================================================================
@dataclass
class TokenCandidate:
    token: str
    chain: Chain
    pool: PoolState
    snapshot: ContractSnapshot
    mentions: pd.Series                   # social mentions per step, DatetimeIndex
    smart_wallet_buys: int = 0


@dataclass
class DiscoveryConfig:
    quote_usd: float = 150.0
    min_liq_usd: float = 10_000.0
    target_liq_usd: float = 250_000.0
    w_velocity: float = 0.30
    w_r0: float = 0.15
    w_attention: float = 0.25
    w_liquidity: float = 0.20
    w_smart: float = 0.10
    buy_threshold: float = 0.55
    watch_threshold: float = 0.30
    source_timeout_s: float = 10.0
    late_penalty: LateBuyerPenalty = field(default_factory=LateBuyerPenalty)


@dataclass
class Suggestion:
    token: str
    chain: Chain
    score: float
    action: str
    components: Dict[str, float]
    safety: SafetyReport


class DiscoverySource(abc.ABC):
    name: str = "source"

    @abc.abstractmethod
    async def fetch_candidates(self) -> List[TokenCandidate]:
        ...


class HttpJsonSource(DiscoverySource):
    """Skeleton for a live DEX API; implement `parse` for your provider's schema."""
    name = "http_json"

    def __init__(self, url: str, retries: int = 3, backoff_s: float = 0.5) -> None:
        self.url, self.retries, self.backoff = url, retries, backoff_s

    def parse(self, payload: Any) -> List[TokenCandidate]:
        raise NotImplementedError

    async def fetch_candidates(self) -> List[TokenCandidate]:
        import aiohttp  # lazy import; pip install aiohttp
        for attempt in range(self.retries):
            try:
                async with aiohttp.ClientSession() as s, s.get(self.url) as r:
                    r.raise_for_status()
                    return self.parse(await r.json())
            except Exception as exc:  # noqa: BLE001
                log.warning("%s attempt %d failed: %s", self.url, attempt + 1, exc)
                await asyncio.sleep(self.backoff * 2 ** attempt)
        return []


class MockDiscoverySource(DiscoverySource):
    name = "mock"

    async def fetch_candidates(self) -> List[TokenCandidate]:
        await asyncio.sleep(0.01)  # simulate network
        out: List[TokenCandidate] = []
        specs = [("EARLY", 400, 0.0), ("LATE", 1500, 0.0), ("HONEY", 450, 0.6)]
        for i, (name, n, sell_tax) in enumerate(specs):
            df = generate_mock_market(n_bars=n, seed=10 + i, freq="5min", delay_bars=40)
            m = df["social_mentions"].resample("15min").sum()
            if name == "EARLY":
                m = m.iloc[:int(np.argmax(m.to_numpy())) - 2]  # still rising
            last = df.iloc[min(len(df), n) - 1]
            holders = [HolderRecord(f"w{j}", 1e6 / (j + 1) ** (2.0 if name == "HONEY" else 0.8),
                                    funder="F1" if name == "HONEY" and j < 6 else None)
                       for j in range(60)]
            snap = ContractSnapshot(
                address=f"0x{i:040x}", chain=Chain.BASE, owner=None, buy_tax_bps=100,
                sell_tax_bps=sell_tax * 1e4, lp_burned_pct=95.0, total_supply=sum(h.balance for h in holders),
                source_code="contract T { function transfer(address a) public {} }",
                holders=holders, sell_simulation_ok=(sell_tax == 0.0))
            pool = PoolState(last["reserve_quote"] / last["close"], last["reserve_quote"], 30.0)
            out.append(TokenCandidate(name, Chain.BASE, pool, snap, m, smart_wallet_buys=2 if name == "EARLY" else 0))
        return out


class DiscoveryEngine:
    def __init__(self, sources: Sequence[DiscoverySource], safety: Optional[SafetyChecker] = None,
                 cfg: Optional[DiscoveryConfig] = None, max_concurrency: int = 8) -> None:
        self.sources = list(sources)
        self.safety = safety or SafetyChecker()
        self.cfg = cfg or DiscoveryConfig()
        self.max_concurrency = max_concurrency

    def score(self, c: TokenCandidate) -> Suggestion:
        cfg = self.cfg
        report = self.safety.check(c.snapshot, c.pool)
        m = c.mentions.astype(float)
        dt_h = max(pd.Series(m.index).diff().dt.total_seconds().median() / 3600.0, 1e-6) if len(m) > 2 else 1.0
        sir = estimate_sir_params(m.to_numpy(), dt_h)
        smooth = m.rolling(3, min_periods=1).mean()
        peak_ts = smooth.idxmax()
        peak = float(smooth.max())
        now = m.index[-1]
        hours_since_peak = max((now - peak_ts).total_seconds() / 3600.0, 0.0)
        att_now = float(smooth.iloc[-1])
        remaining = att_now / peak if peak > 0 else 0.0
        k = min(3, len(smooth) - 1)
        growth = (math.log(max(att_now, 1e-9)) - math.log(max(float(smooth.iloc[-1 - k]), 1e-9))) / (k * dt_h) if k else 0.0
        vel_score = 0.5 * (math.tanh(growth) + 1.0)
        liq_usd = 2.0 * c.pool.reserve_quote * cfg.quote_usd
        liq_score = float(np.clip(math.log10(max(liq_usd, 1.0) / cfg.min_liq_usd) /
                                  math.log10(cfg.target_liq_usd / cfg.min_liq_usd), 0.0, 1.0))
        r0_score = float(np.clip((sir.r0 - 1.0) / 3.0, 0.0, 1.0))
        smart = float(np.clip(c.smart_wallet_buys / 3.0, 0.0, 1.0))
        base = (cfg.w_velocity * vel_score + cfg.w_r0 * r0_score + cfg.w_attention * remaining
                + cfg.w_liquidity * liq_score + cfg.w_smart * smart)
        late = cfg.late_penalty.multiplier(hours_since_peak)
        final = base * late if report.passed else 0.0
        action = "AVOID" if not report.passed else (
            "BUY" if final >= cfg.buy_threshold else "WATCH" if final >= cfg.watch_threshold else "SKIP")
        comps = {"base": base, "late_multiplier": late, "hours_since_peak": hours_since_peak,
                 "velocity": vel_score, "r0": sir.r0, "attention_remaining": remaining,
                 "liquidity": liq_score, "liquidity_usd": liq_usd, "half_life_h": sir.half_life_hours,
                 "attention_1h_ahead": attention_decay(att_now, 1.0, sir.half_life_hours)}
        return Suggestion(c.token, c.chain, final, action, comps, report)

    async def poll_once(self) -> List[Suggestion]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def guarded(src: DiscoverySource) -> List[TokenCandidate]:
            async with sem:
                try:
                    return await asyncio.wait_for(src.fetch_candidates(), self.cfg.source_timeout_s)
                except Exception as exc:  # noqa: BLE001
                    log.warning("source %s failed: %s", src.name, exc)
                    return []
        batches = await asyncio.gather(*(guarded(s) for s in self.sources))
        uniq = {(c.chain, c.token): c for c in ichain.from_iterable(batches)}
        return sorted((self.score(c) for c in uniq.values()), key=lambda s: s.score, reverse=True)

    async def run_forever(self, interval_s: float,
                          on_update: Callable[[List[Suggestion]], Awaitable[None]],
                          max_cycles: Optional[int] = None) -> None:
        n = 0
        while max_cycles is None or n < max_cycles:
            await on_update(await self.poll_once())
            n += 1
            await asyncio.sleep(interval_s)


# =============================================================================
# 9. Demo
# =============================================================================
def demo() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = CHAINS[Chain.SOLANA]
    ing = DataIngestion(MockDataSource({"AAA": dict(seed=1), "BBB": dict(seed=2, rug_at=1500)}))
    frames = asyncio.run(ing.load_many(["AAA", "BBB"]))

    print("== Safety check ==")
    sketchy = ContractSnapshot(
        "0xabc", Chain.ETHEREUM, owner="0x1234567890123456789012345678901234567890",
        source_code="""contract T { mapping(address=>bool) bl;
            function setBlacklist(address a,bool v) external onlyOwner { bl[a]=v; }
            function setTaxes(uint b,uint s) external onlyOwner { _sellTax = s; }
            function enableTrading() external onlyOwner { tradingOpen = true; } }""",
        sell_tax_bps=1200, lp_locked_pct=0, total_supply=1e9,
        holders=[HolderRecord(f"w{i}", 8e7, funder="F") for i in range(6)] +
                [HolderRecord(f"x{i}", 1e5) for i in range(50)])
    pool = PoolState(1e9, 30.0, 30.0)
    rep = SafetyChecker().check(sketchy, pool)
    print(f"passed={rep.passed} risk={rep.risk_score:.0f}", [f.code for f in rep.flags])

    print("\n== Vectorized sweep (numba=%s) ==" % HAS_NUMBA)
    vb = VectorizedBacktester(cfg, buy_tax_bps=100, sell_tax_bps=100, marketing_bps=50,
                              size_quote=1.0, init_cash=10.0, stop_loss=0.3, trail=0.2)
    print(vb.sweep(frames["AAA"], EmaMomentumStrategy,
                   {"fast": [3, 5, 8], "slow": [15, 30, 60]}).head(5).round(3).to_string())

    print("\n== Event-driven (trailing stops, 2 tokens, rug on BBB) ==")
    sim = AMMExecutionSim(cfg, buy_tax_bps=100, sell_tax_bps=100, marketing_fee_bps=50)
    feed = DataIngestion.to_event_feed(frames)
    eng = StrategyEngine()
    res = EventDrivenBacktester(sim, init_cash=10.0).run(
        feed, eng.event("trailing_momentum", lookback=30, size_quote=1.0))
    print({k: round(v, 3) for k, v in res.metrics.items()}, "trades:", len(res.trades))

    print("\n== Discovery ==")
    async def show(sugs: List[Suggestion]) -> None:
        for s in sugs:
            print(f"{s.token:6s} {s.action:6s} score={s.score:.2f} late_x={s.components['late_multiplier']:.2f} "
                  f"h_since_peak={s.components['hours_since_peak']:.1f} risk={s.safety.risk_score:.0f}")
    asyncio.run(DiscoveryEngine([MockDiscoverySource()]).run_forever(0.0, show, max_cycles=1))


if __name__ == "__main__":
    demo()
''',
    'dexscreener_adapter': r'''"""
dexscreener_adapter.py -- live DexScreener data for memecoin_quant.py

Plugs into the existing ABCs:
    DexScreenerDiscoverySource  -> DiscoverySource   (feeds DiscoveryEngine)
    RecordedCsvSource           -> MarketDataSource  (feeds DataIngestion / backtesters)
    DexScreenerRecorder         -> builds the history DexScreener does not provide

What the API does NOT give you (so the adapter works around it):
    * No historical candles  -> DexScreenerRecorder snapshots pairs on an interval to CSV.
    * No contract/safety data -> SafetyDataProvider ABC (wire RugCheck/GoPlus/etc. yourself).
    * No social mentions     -> txn counts are used as an attention proxy (see activity_series).

Endpoints used (base https://api.dexscreener.com, no key):
    GET /token-profiles/latest/v1, /token-boosts/latest/v1, /token-boosts/top/v1   (60 req/min)
    GET /tokens/v1/{chainId}/{addr1,addr2,...}, /token-pairs/v1/{chainId}/{addr},
        /latest/dex/search?q=                                                       (~300 req/min)
Limiters run at 90% of those figures. The per-call token batch cap (30) is from memory:
verify it against the current docs and adjust DexScreenerClient.MAX_TOKENS_PER_CALL.
"""
from __future__ import annotations

import abc
import asyncio
import csv
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from memecoin_quant import (CHAINS, Chain, ContractSnapshot, DataIngestion, DiscoverySource,
                            MarketDataSource, PoolState, TokenCandidate)

log = logging.getLogger("dexscreener_adapter")

BASE_URL = "https://api.dexscreener.com"
NATIVE_SYMBOLS = {"SOL", "WSOL", "ETH", "WETH", "BNB", "WBNB"}
# Pools whose reserves are not plain x*y=k. Tune from the labels you actually observe.
CP_LABEL_BLOCKLIST = {"CLMM", "V3", "V4", "DLMM", "WP"}


# =============================================================================
# Errors, rate limiting, transports
# =============================================================================
class RateLimited(Exception):
    def __init__(self, retry_after: Optional[float] = None) -> None:
        super().__init__(f"rate limited (retry_after={retry_after})")
        self.retry_after = retry_after


class HttpError(Exception):
    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status == 0 or self.status >= 500


class AsyncRateLimiter:
    """Sliding-window limiter: at most `per_minute` acquisitions in any 60s."""

    def __init__(self, per_minute: int) -> None:
        self.n = max(1, per_minute)
        self._stamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.n:
                    self._stamps.append(now)
                    return
                await asyncio.sleep(60.0 - (now - self._stamps[0]) + 0.01)


class Transport(abc.ABC):
    @abc.abstractmethod
    async def get_json(self, path: str, params: Optional[Mapping[str, str]] = None) -> Any:
        """Raise RateLimited on 429, HttpError otherwise."""

    async def close(self) -> None:  # pragma: no cover
        return None


class AiohttpTransport(Transport):
    def __init__(self, base_url: str = BASE_URL, timeout_s: float = 10.0) -> None:
        self.base, self.timeout_s = base_url, timeout_s
        self._session: Any = None

    async def _sess(self) -> Any:
        if self._session is None:
            import aiohttp  # pip install aiohttp
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        return self._session

    async def get_json(self, path: str, params: Optional[Mapping[str, str]] = None) -> Any:
        import aiohttp
        s = await self._sess()
        try:
            async with s.get(self.base + path, params=params) as r:
                if r.status == 429:
                    ra = r.headers.get("Retry-After", "")
                    raise RateLimited(float(ra) if ra.replace(".", "", 1).isdigit() else None)
                if r.status >= 400:
                    raise HttpError(r.status, await r.text())
                return await r.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise HttpError(0, str(exc)) from exc

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class FixtureTransport(Transport):
    """Offline transport for tests: routes maps a path prefix to JSON or a callable(path, params)."""

    def __init__(self, routes: Mapping[str, Any]) -> None:
        self.routes = dict(routes)
        self.calls: List[str] = []

    async def get_json(self, path: str, params: Optional[Mapping[str, str]] = None) -> Any:
        self.calls.append(path)
        for prefix in sorted(self.routes, key=len, reverse=True):
            if path.startswith(prefix):
                v = self.routes[prefix]
                return v(path, params) if callable(v) else v
        raise HttpError(404, path)


# =============================================================================
# Parsed models
# =============================================================================
def _f(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _norm(a: str) -> str:
    return a.lower() if a.startswith("0x") else a


def _same(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


@dataclass(frozen=True)
class TokenRef:
    chain_id: str
    address: str
    source: str


@dataclass(frozen=True)
class Oriented:
    """A pair viewed from the tracked token's side."""
    price: float            # quote per token
    reserve_token: float
    reserve_quote: float
    counter_symbol: str


@dataclass
class PairSnapshot:
    chain_id: str
    dex_id: str
    pair_address: str
    labels: List[str]
    base_addr: str
    base_symbol: str
    quote_addr: str
    quote_symbol: str
    price_native: float
    price_usd: Optional[float]
    liq_usd: Optional[float]
    liq_base: Optional[float]
    liq_quote: Optional[float]
    txns: Dict[str, Tuple[int, int]]       # window -> (buys, sells)
    volume: Dict[str, float]
    price_change: Dict[str, float]
    fdv: Optional[float]
    market_cap: Optional[float]
    created_ms: Optional[int]
    boosts_active: int = 0

    def txn_count(self, window: str) -> int:
        b, s = self.txns.get(window, (0, 0))
        return b + s

    def age_hours(self, now_ms: Optional[int] = None) -> Optional[float]:
        if self.created_ms is None:
            return None
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        return max((now_ms - self.created_ms) / 3.6e6, 0.0)

    def orient(self, token: str) -> Optional[Oriented]:
        b, q = self.liq_base, self.liq_quote
        if not b or not q or b <= 0 or q <= 0 or self.price_native <= 0:
            return None
        if _same(token, self.base_addr):
            return Oriented(self.price_native, b, q, self.quote_symbol)
        if _same(token, self.quote_addr):
            return Oriented(1.0 / self.price_native, q, b, self.base_symbol)
        return None


def parse_pair(raw: Mapping[str, Any]) -> Optional[PairSnapshot]:
    """Tolerant parser: returns None for malformed rows instead of raising."""
    try:
        pn = _f(raw.get("priceNative"))
        if pn is None or pn <= 0:
            return None
        liq = raw.get("liquidity") or {}
        base, quote = raw.get("baseToken") or {}, raw.get("quoteToken") or {}
        txns = {k: (int(v.get("buys", 0) or 0), int(v.get("sells", 0) or 0))
                for k, v in (raw.get("txns") or {}).items()}
        return PairSnapshot(
            chain_id=str(raw["chainId"]), dex_id=str(raw.get("dexId", "")),
            pair_address=str(raw["pairAddress"]), labels=list(raw.get("labels") or []),
            base_addr=str(base["address"]), base_symbol=str(base.get("symbol") or "?"),
            quote_addr=str(quote.get("address") or ""), quote_symbol=str(quote.get("symbol") or "?"),
            price_native=pn, price_usd=_f(raw.get("priceUsd")),
            liq_usd=_f(liq.get("usd")), liq_base=_f(liq.get("base")), liq_quote=_f(liq.get("quote")),
            txns=txns, volume={k: float(v) for k, v in (raw.get("volume") or {}).items() if _f(v) is not None},
            price_change={k: float(v) for k, v in (raw.get("priceChange") or {}).items() if _f(v) is not None},
            fdv=_f(raw.get("fdv")), market_cap=_f(raw.get("marketCap")),
            created_ms=int(raw["pairCreatedAt"]) if raw.get("pairCreatedAt") else None,
            boosts_active=int((raw.get("boosts") or {}).get("active", 0) or 0))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _as_list(payload: Any, key: str = "pairs") -> List[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        v = payload.get(key)
        return v if isinstance(v, list) else [payload]
    return []


def best_pair(pairs: Sequence[PairSnapshot], token: str, require_native_quote: bool = True,
              blocklist: Set[str] = frozenset(CP_LABEL_BLOCKLIST),
              max_curve_deviation: float = 0.10) -> Optional[PairSnapshot]:
    """Highest-liquidity pair that our x*y=k simulator can honestly model."""
    block = {b.upper() for b in blocklist}
    best: Optional[PairSnapshot] = None
    for p in pairs:
        o = p.orient(token)
        if o is None:
            continue
        if require_native_quote and o.counter_symbol.upper() not in NATIVE_SYMBOLS:
            continue
        if {l.upper() for l in p.labels} & block:
            continue
        if abs((o.reserve_quote / o.reserve_token) / o.price - 1.0) > max_curve_deviation:
            continue  # reserves inconsistent with price => not a plain constant-product pool
        if best is None or (p.liq_usd or 0.0) > (best.liq_usd or 0.0):
            best = p
    return best


def activity_series(pair: PairSnapshot, now: pd.Timestamp) -> pd.Series:
    """
    Attention proxy (DexScreener has no social data): hourly txn counts for the last 24h,
    rebuilt from the m5/h1/h6/h24 windows. Hour 0 = h1 exactly; hours 1-5 and 6-23 are flat
    averages of their windows, zeroed before pair creation. Plateaus make peak timing coarse;
    prefer a recorded series (DexScreenerRecorder) once you have history.
    """
    h1 = pair.txn_count("h1")
    h6 = max(pair.txn_count("h6"), h1)
    h24 = max(pair.txn_count("h24"), h6)
    age = pair.age_hours()
    blocks = [(0, 0, h1), (1, 5, h6 - h1), (6, 23, h24 - h6)]
    vals = [0.0] * 24                       # index = hours_ago
    for lo, hi, total in blocks:
        valid = [k for k in range(lo, hi + 1) if lo == 0 or age is None or k <= age]
        for k in valid:
            vals[k] = total / len(valid)
    idx = pd.date_range(end=now, periods=24, freq="1h")
    return pd.Series(list(reversed(vals)), index=idx, name=pair.base_symbol)


# =============================================================================
# Client
# =============================================================================
class DexScreenerClient:
    MAX_TOKENS_PER_CALL = 30  # verify against current docs

    def __init__(self, transport: Optional[Transport] = None, profile_rpm: int = 54,
                 pairs_rpm: int = 270, max_retries: int = 4, backoff_s: float = 1.0) -> None:
        self.transport = transport or AiohttpTransport()
        self._lim = {"profiles": AsyncRateLimiter(profile_rpm), "pairs": AsyncRateLimiter(pairs_rpm)}
        self.max_retries, self.backoff = max_retries, backoff_s

    async def _get(self, path: str, params: Optional[Mapping[str, str]] = None,
                   group: str = "pairs") -> Any:
        for attempt in range(self.max_retries + 1):
            await self._lim[group].acquire()
            try:
                return await self.transport.get_json(path, params)
            except RateLimited as exc:
                delay, err = exc.retry_after or self.backoff * 2 ** attempt, exc
            except HttpError as exc:
                if not exc.retryable:
                    raise
                delay, err = self.backoff * 2 ** attempt, exc
            except asyncio.TimeoutError as exc:
                delay, err = self.backoff * 2 ** attempt, exc
            if attempt == self.max_retries:
                raise err
            log.warning("%s failed (%s); retry in %.1fs", path, err, delay)
            await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    async def _refs(self, path: str, source: str) -> List[TokenRef]:
        data = await self._get(path, group="profiles")
        return [TokenRef(str(d["chainId"]), str(d["tokenAddress"]), source)
                for d in _as_list(data) if isinstance(d, dict) and d.get("chainId") and d.get("tokenAddress")]

    async def latest_profiles(self) -> List[TokenRef]:
        return await self._refs("/token-profiles/latest/v1", "profiles")

    async def latest_boosts(self) -> List[TokenRef]:
        return await self._refs("/token-boosts/latest/v1", "boosts")

    async def top_boosts(self) -> List[TokenRef]:
        return await self._refs("/token-boosts/top/v1", "top_boosts")

    async def token_pairs(self, chain: str, address: str) -> List[PairSnapshot]:
        data = await self._get(f"/token-pairs/v1/{chain}/{address}")
        return [p for p in map(parse_pair, _as_list(data)) if p]

    async def search(self, query: str) -> List[PairSnapshot]:
        data = await self._get("/latest/dex/search", {"q": query})
        return [p for p in map(parse_pair, _as_list(data)) if p]

    async def tokens(self, chain: str, addresses: Sequence[str]) -> List[PairSnapshot]:
        addrs = list(dict.fromkeys(addresses))
        n = self.MAX_TOKENS_PER_CALL
        chunks = [addrs[i:i + n] for i in range(0, len(addrs), n)]
        res = await asyncio.gather(*(self._get(f"/tokens/v1/{chain}/{','.join(c)}") for c in chunks),
                                   return_exceptions=True)
        out: List[PairSnapshot] = []
        for c, r in zip(chunks, res):
            if isinstance(r, Exception):
                log.warning("tokens batch (%d addrs) failed: %s", len(c), r)
                continue
            out.extend(p for p in map(parse_pair, _as_list(r)) if p)
        return out

    async def close(self) -> None:
        await self.transport.close()


# =============================================================================
# Safety data provider (DexScreener has none)
# =============================================================================
class SafetyDataProvider(abc.ABC):
    @abc.abstractmethod
    async def snapshot(self, chain: Chain, token: str) -> ContractSnapshot:
        """Fetch owner/authorities/source/taxes/LP-lock/holders from RugCheck, GoPlus, an RPC, etc."""


class NullSafetyProvider(SafetyDataProvider):
    """Fails closed: an empty snapshot trips UNVERIFIED_SOURCE + LP_UNLOCKED, so nothing is a BUY."""

    async def snapshot(self, chain: Chain, token: str) -> ContractSnapshot:
        return ContractSnapshot(address=token, chain=chain)


class StaticSafetyProvider(SafetyDataProvider):
    def __init__(self, by_token: Mapping[str, ContractSnapshot],
                 default: Optional[ContractSnapshot] = None) -> None:
        self.by_token, self.default = dict(by_token), default

    async def snapshot(self, chain: Chain, token: str) -> ContractSnapshot:
        s = self.by_token.get(token) or self.default
        return s if s is not None else ContractSnapshot(address=token, chain=chain)


# =============================================================================
# Discovery source
# =============================================================================
@dataclass
class DexScreenerSourceConfig:
    chains: Set[str] = field(default_factory=lambda: {"solana", "base"})
    use_profiles: bool = True
    use_boosts: bool = True
    use_top_boosts: bool = False
    min_liquidity_usd: float = 10_000.0
    min_pair_age_hours: float = 0.0
    max_pair_age_hours: float = 72.0
    max_candidates: int = 60
    require_native_quote: bool = True
    max_curve_deviation: float = 0.10
    cp_label_blocklist: Set[str] = field(default_factory=lambda: set(CP_LABEL_BLOCKLIST))
    fee_bps_by_dex: Dict[str, float] = field(default_factory=dict)  # e.g. {"pumpswap": 25.0}


class DexScreenerDiscoverySource(DiscoverySource):
    name = "dexscreener"

    def __init__(self, client: DexScreenerClient, safety: Optional[SafetyDataProvider] = None,
                 cfg: Optional[DexScreenerSourceConfig] = None) -> None:
        self.client = client
        self.safety = safety or NullSafetyProvider()
        self.cfg = cfg or DexScreenerSourceConfig()
        self.last_pairs: Dict[str, PairSnapshot] = {}   # candidate key -> raw pair (for lookups)

    async def _collect_refs(self) -> List[Tuple[str, str]]:
        cfg, calls = self.cfg, []
        if cfg.use_profiles:
            calls.append(self.client.latest_profiles())
        if cfg.use_boosts:
            calls.append(self.client.latest_boosts())
        if cfg.use_top_boosts:
            calls.append(self.client.top_boosts())
        seen: Dict[Tuple[str, str], None] = {}
        for r in await asyncio.gather(*calls, return_exceptions=True):
            if isinstance(r, Exception):
                log.warning("ref endpoint failed: %s", r)
                continue
            for ref in r:
                if ref.chain_id in cfg.chains:
                    seen.setdefault((ref.chain_id, ref.address), None)
        return list(seen)[: cfg.max_candidates]

    async def fetch_candidates(self) -> List[TokenCandidate]:
        cfg = self.cfg
        keys = await self._collect_refs()
        by_chain: Dict[str, List[str]] = defaultdict(list)
        for ch, a in keys:
            by_chain[ch].append(a)
        chains = list(by_chain)
        pair_lists = await asyncio.gather(*(self.client.tokens(ch, by_chain[ch]) for ch in chains))
        now_ms = int(time.time() * 1000)
        now_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)

        picked: List[Tuple[str, str, PairSnapshot]] = []
        for ch, pairs in zip(chains, pair_lists):
            index: Dict[str, List[PairSnapshot]] = defaultdict(list)
            for p in pairs:
                index[_norm(p.base_addr)].append(p)
                index[_norm(p.quote_addr)].append(p)
            for a in by_chain[ch]:
                bp = best_pair(index.get(_norm(a), []), a, cfg.require_native_quote,
                               cfg.cp_label_blocklist, cfg.max_curve_deviation)
                if bp is None or (bp.liq_usd or 0.0) < cfg.min_liquidity_usd:
                    continue
                age = bp.age_hours(now_ms)
                if age is not None and not (cfg.min_pair_age_hours <= age <= cfg.max_pair_age_hours):
                    continue
                picked.append((ch, a, bp))

        snaps = await asyncio.gather(*(self.safety.snapshot(Chain(ch), a) for ch, a, _ in picked),
                                     return_exceptions=True)
        out: List[TokenCandidate] = []
        for (ch, a, bp), snap in zip(picked, snaps):
            chain = Chain(ch)
            if isinstance(snap, Exception):
                log.warning("safety provider failed for %s: %s", a, snap)
                snap = ContractSnapshot(address=a, chain=chain)
            o = bp.orient(a)
            if o is None:
                continue
            fee = cfg.fee_bps_by_dex.get(bp.dex_id, CHAINS[chain].pool_fee_bps)
            key = f"{bp.base_symbol if _same(a, bp.base_addr) else bp.quote_symbol}:{a[:8]}"
            self.last_pairs[key] = bp
            out.append(TokenCandidate(key, chain, PoolState(o.reserve_token, o.reserve_quote, fee),
                                      snap, activity_series(bp, now_ts)))
        return out


# =============================================================================
# Recorder: build your own history (DexScreener has no candles)
# =============================================================================
CSV_FIELDS = ["timestamp", "close", "volume", "reserve_quote", "reserve_token",
              "social_mentions", "congestion", "liquidity_usd"]


class SnapshotStore:
    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, chain: str, token: str) -> Path:
        return self.root / f"{chain}_{token}.csv"

    def append(self, chain: str, token: str, ts: pd.Timestamp, pair: PairSnapshot, o: Oriented) -> None:
        p = self.path(chain, token)
        new = not p.exists()
        with p.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(CSV_FIELDS)
            w.writerow([ts.isoformat(), o.price, pair.volume.get("m5", 0.0), o.reserve_quote,
                        o.reserve_token, pair.txn_count("m5"), 0.0, pair.liq_usd or 0.0])

    def load(self, chain: str, token: str) -> pd.DataFrame:
        df = pd.read_csv(self.path(chain, token), parse_dates=["timestamp"])
        return df.set_index("timestamp").sort_index()


class RecordedCsvSource(MarketDataSource):
    """MarketDataSource over recorded snapshots. `volume` is m5 USD volume; `social_mentions`
    is m5 txn count. Set bars_per_year = 365*24*3600 / interval_s in the backtesters."""

    def __init__(self, store: SnapshotStore, chain: str) -> None:
        self.store, self.chain = store, chain

    def load(self, token: str) -> pd.DataFrame:
        return self.store.load(self.chain, token)


class DexScreenerRecorder:
    def __init__(self, client: DexScreenerClient, store: SnapshotStore,
                 tokens: Sequence[Tuple[str, str]], cfg: Optional[DexScreenerSourceConfig] = None) -> None:
        self.client, self.store, self.tokens = client, store, list(tokens)
        self.cfg = cfg or DexScreenerSourceConfig()

    async def record_once(self) -> int:
        by_chain: Dict[str, List[str]] = defaultdict(list)
        for ch, a in self.tokens:
            by_chain[ch].append(a)
        chains = list(by_chain)
        lists = await asyncio.gather(*(self.client.tokens(ch, by_chain[ch]) for ch in chains))
        ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
        n = 0
        for ch, pairs in zip(chains, lists):
            index: Dict[str, List[PairSnapshot]] = defaultdict(list)
            for p in pairs:
                index[_norm(p.base_addr)].append(p)
                index[_norm(p.quote_addr)].append(p)
            for a in by_chain[ch]:
                bp = best_pair(index.get(_norm(a), []), a, self.cfg.require_native_quote,
                               self.cfg.cp_label_blocklist, self.cfg.max_curve_deviation)
                o = bp.orient(a) if bp else None
                if bp is None or o is None:
                    log.warning("no modelable pair for %s:%s this cycle", ch, a)
                    continue
                self.store.append(ch, a, ts, bp, o)
                n += 1
        return n

    async def run(self, interval_s: float = 30.0, max_cycles: Optional[int] = None) -> None:
        c = 0
        while max_cycles is None or c < max_cycles:
            log.info("recorded %d snapshots", await self.record_once())
            c += 1
            await asyncio.sleep(interval_s)


# =============================================================================
# Offline demo (no network needed)
# =============================================================================
def _mk_pair(chain: str, addr: str, sym: str, quote_sym: str = "SOL", price: float = 1e-6,
             liq_quote: float = 50.0, labels: Optional[List[str]] = None, age_h: float = 5.0,
             txn_scale: int = 1, price_mult: float = 1.0, reserve_skew: float = 1.0) -> Dict[str, Any]:
    p = price * price_mult
    liq_quote = liq_quote * price_mult ** 0.5            # x*y=k: price up => quote reserve up
    base_res = liq_quote / p * reserve_skew              # token reserve consistent with price
    now_ms = int(time.time() * 1000)
    t = lambda b, s: {"buys": b * txn_scale, "sells": s * txn_scale}   # noqa: E731
    return {
        "chainId": chain, "dexId": "raydium", "pairAddress": f"pair_{sym}", "labels": labels or [],
        "baseToken": {"address": addr, "name": sym, "symbol": sym},
        "quoteToken": {"address": "So11111111111111111111111111111111111111112", "name": quote_sym,
                       "symbol": quote_sym},
        "priceNative": f"{p:.12f}", "priceUsd": f"{p * 150:.12f}",
        "txns": {"m5": t(40, 25), "h1": t(300, 220), "h6": t(900, 800), "h24": t(1500, 1400)},
        "volume": {"m5": 5000.0, "h1": 60000.0}, "priceChange": {"m5": 2.0, "h1": 15.0},
        "liquidity": {"usd": liq_quote * 2 * 150, "base": base_res, "quote": liq_quote},
        "fdv": 1e6, "marketCap": 1e6, "pairCreatedAt": int(now_ms - age_h * 3.6e6),
        "boosts": {"active": 1}}


def _demo() -> None:
    from memecoin_quant import (DataIngestion, DiscoveryEngine, EmaMomentumStrategy, Side,
                                VectorizedBacktester)
    logging.basicConfig(level=logging.INFO)
    A, B, C, D = "AAAA1111", "BBBB2222", "CCCC3333", "DDDD4444"
    tick = {"n": 0}

    def tokens_route(path: str, params: Any) -> List[Dict[str, Any]]:
        tick["n"] += 1
        drift = 1.0 + 0.02 * tick["n"]
        return [
            _mk_pair("solana", A, "AAA", price_mult=drift),                       # clean, recordable
            _mk_pair("solana", B, "CLMMCOIN", labels=["CLMM"]),                   # filtered: CLMM
            _mk_pair("solana", C, "USDCPAIR", quote_sym="USDC"),                  # filtered: non-native quote
            _mk_pair("solana", D, "SKEWED", reserve_skew=3.0),                    # filtered: curve != price
        ]

    routes = {
        "/token-profiles/latest/v1": [{"chainId": "solana", "tokenAddress": a} for a in (A, B, C)],
        "/token-boosts/latest/v1": [{"chainId": "solana", "tokenAddress": a} for a in (A, D)]
                                   + [{"chainId": "ton", "tokenAddress": "x"}],
        "/tokens/v1/": tokens_route,
    }
    client = DexScreenerClient(FixtureTransport(routes))
    clean = ContractSnapshot("x", Chain.SOLANA, lp_burned_pct=100.0, total_supply=1e9,
                             source_code="contract T { function transfer(address a) public {} }",
                             sell_simulation_ok=True)
    src = DexScreenerDiscoverySource(client, StaticSafetyProvider({}, default=clean),
                                     DexScreenerSourceConfig(min_liquidity_usd=5_000))
    cands = asyncio.run(src.fetch_candidates())
    print("candidates:", [c.token for c in cands])
    for s in asyncio.run(DiscoveryEngine([src]).poll_once()):
        print(f"{s.token:16s} {s.action:6s} score={s.score:.2f} risk={s.safety.risk_score:.0f} "
              f"late_x={s.components['late_multiplier']:.2f}")

    import tempfile
    store = SnapshotStore(tempfile.mkdtemp())
    rec = DexScreenerRecorder(client, store, [("solana", A)])
    asyncio.run(rec.run(interval_s=0.0, max_cycles=5))
    frame = DataIngestion(RecordedCsvSource(store, "solana")).load(A)
    print(frame[["close", "reserve_quote", "social_mentions"]].round(9).to_string())


if __name__ == "__main__":
    _demo()
''',
    'safety_module': r'''"""
safety_module.py -- fail-closed pre-trade token security checks.

    SafetyDataProvider (ABC)  async is_safe(token_address, chain) -> bool
    ├── FailClosedProvider       always False (safe default)
    ├── RugCheckProvider         Solana SPL tokens   (api.rugcheck.xyz)
    ├── GoPlusProvider           EVM: Ethereum/BSC/Base/Arbitrum (api.gopluslabs.io)
    └── CompositeSafetyProvider  routes by chain

Fail-closed means False on: timeouts, network errors, HTTP errors (incl. 429), non-JSON,
unknown chains, malformed addresses, and ANY missing/unparseable field a rule depends on.
A null authority is a valid value (renounced); an ABSENT authority key is not.

Run:  python safety_module.py          (offline mock harness, no network)
      python safety_module.py --live   (also hits the real APIs)

Notes: response field names were taken from public docs/examples and could not be
verified against the live APIs from the build sandbox. Parsing is isolated in
RugCheckProvider._check / _lp_secure_pct and GoPlusProvider._check so they are easy to adjust.
"""
from __future__ import annotations

import abc
import argparse
import asyncio
import copy
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import aiohttp

log = logging.getLogger("safety_module")

_EVM_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SOL_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")  # base58, no 0/O/I/l

CHAIN_ALIASES: Dict[str, str] = {
    "solana": "solana", "sol": "solana",
    "ethereum": "ethereum", "eth": "ethereum",
    "bsc": "bsc", "bnb": "bsc",
    "base": "base",
    "arbitrum": "arbitrum", "arb": "arbitrum",
}
GOPLUS_CHAIN_IDS: Dict[str, int] = {"ethereum": 1, "bsc": 56, "base": 8453, "arbitrum": 42161}


def normalize_chain(chain: Any) -> Optional[str]:
    return CHAIN_ALIASES.get(str(chain).strip().lower())


def _num(x: Any) -> Optional[float]:
    """float(x) for int/float/numeric-string; None for bool, empty, junk, NaN."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


# =============================================================================
# Verdict + HTTP layer
# =============================================================================
@dataclass(frozen=True)
class SafetyVerdict:
    safe: bool
    provider: str
    reasons: Tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)   # raw values the checks used

    @classmethod
    def ok(cls, provider: str, metrics: Optional[Mapping[str, Any]] = None) -> "SafetyVerdict":
        return cls(True, provider, (), dict(metrics or {}))

    @classmethod
    def fail(cls, provider: str, *reasons: str,
             metrics: Optional[Mapping[str, Any]] = None) -> "SafetyVerdict":
        return cls(False, provider, tuple(reasons) or ("failed",), dict(metrics or {}))


class FetchError(Exception):
    """Transport-level failure. status 0 = timeout / connection / bad JSON."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(f"HTTP {status}: {message}" if status else message)
        self.status = status


class JsonFetcher(abc.ABC):
    @abc.abstractmethod
    async def get_json(self, url: str, headers: Optional[Mapping[str, str]] = None) -> Any:
        """Return parsed JSON for HTTP 200; raise FetchError otherwise."""

    async def aclose(self) -> None:
        return None


class AiohttpFetcher(JsonFetcher):
    def __init__(self, timeout_s: float = 5.0,
                 session: Optional[aiohttp.ClientSession] = None) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session = session
        self._owns = session is None

    async def get_json(self, url: str, headers: Optional[Mapping[str, str]] = None) -> Any:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(url, headers=headers, timeout=self._timeout) as r:
                if r.status != 200:
                    raise FetchError(r.status, (await r.text())[:200])
                return await r.json(content_type=None)
        except asyncio.TimeoutError as exc:
            raise FetchError(0, "timeout") from exc
        except aiohttp.ClientError as exc:
            raise FetchError(0, f"client error: {exc}") from exc
        except ValueError as exc:
            raise FetchError(0, "invalid JSON") from exc

    async def aclose(self) -> None:
        if self._owns and self._session is not None:
            await self._session.close()
            self._session = None


# =============================================================================
# ABC + providers
# =============================================================================
class SafetyDataProvider(abc.ABC):
    @abc.abstractmethod
    async def is_safe(self, token_address: str, chain: str) -> bool:
        """True ONLY if every check passes. Must never raise; any doubt => False."""


class FailClosedProvider(SafetyDataProvider):
    """Default: nothing is safe until a real provider is wired in."""

    async def is_safe(self, token_address: str, chain: str) -> bool:
        return False


class _HttpProvider(SafetyDataProvider):
    """Shared plumbing: validation, caching, concurrency cap, and the fail-closed wrapper."""
    name = "http"

    def __init__(self, fetcher: JsonFetcher, cache_ttl_s: float = 300.0,
                 overall_timeout_s: float = 8.0, max_concurrency: int = 4) -> None:
        self.fetcher = fetcher
        self.cache_ttl_s = cache_ttl_s
        self.overall_timeout_s = overall_timeout_s
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache: Dict[Tuple[str, str], Tuple[float, SafetyVerdict]] = {}

    @abc.abstractmethod
    async def _check(self, token_address: str, chain: str) -> SafetyVerdict:
        """Raise FetchError on transport problems; return a verdict for any definitive answer."""

    async def evaluate(self, token_address: str, chain: str) -> SafetyVerdict:
        norm = normalize_chain(chain)
        if norm is None:
            return SafetyVerdict.fail(self.name, f"unknown chain {chain!r}")
        addr = str(token_address).strip()
        key = (norm, addr.lower() if addr.startswith("0x") else addr)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl_s:
            return hit[1]
        try:
            async with self._sem:
                verdict = await asyncio.wait_for(self._check(addr, norm), self.overall_timeout_s)
        except FetchError as exc:
            return SafetyVerdict.fail(self.name, f"fetch failed: {exc}")   # not cached
        except asyncio.TimeoutError:
            return SafetyVerdict.fail(self.name, "timeout")                # not cached
        except Exception as exc:  # noqa: BLE001 -- fail closed on anything unexpected
            log.exception("unexpected error checking %s", addr)
            return SafetyVerdict.fail(self.name, f"unexpected error: {type(exc).__name__}")
        self._cache[key] = (time.monotonic(), verdict)                     # only definitive answers
        return verdict

    async def is_safe(self, token_address: str, chain: str) -> bool:
        return (await self.evaluate(token_address, chain)).safe


_MISSING = object()


class RugCheckProvider(_HttpProvider):
    """Solana SPL tokens via GET https://api.rugcheck.xyz/v1/tokens/{mint}/report"""
    name = "rugcheck"

    def __init__(self, fetcher: JsonFetcher, max_score: float = 5000.0, min_lp_secure_pct: float = 80.0,
                 base_url: str = "https://api.rugcheck.xyz", **kw: Any) -> None:
        super().__init__(fetcher, **kw)
        self.max_score, self.min_lp, self.base_url = max_score, min_lp_secure_pct, base_url.rstrip("/")

    @staticmethod
    def _authority(report: Mapping[str, Any], camel: str, snake: str) -> Any:
        for src in (report, report.get("token") if isinstance(report.get("token"), dict) else {}):
            for k in (camel, snake):
                if k in src:
                    return src[k]
        return _MISSING

    @staticmethod
    def _lp_secure_pct(report: Mapping[str, Any]) -> Optional[float]:
        """Liquidity-weighted (locked + burned)% across markets, capped at 100 per market.
        `lpBurnedPct` is a best-effort field name; if RugCheck already folds burned LP into
        lpLockedPct, the cap prevents overcounting beyond 100 but you should verify."""
        markets = report.get("markets")
        if not isinstance(markets, list):
            return None
        rows: List[Tuple[float, float]] = []
        for m in markets:
            lp = m.get("lp") if isinstance(m, dict) else None
            if not isinstance(lp, dict):
                continue
            locked = _num(lp.get("lpLockedPct"))
            if locked is None:
                continue
            burned = _num(lp.get("lpBurnedPct")) or 0.0
            weight = (_num(lp.get("baseUSD")) or 0.0) + (_num(lp.get("quoteUSD")) or 0.0)
            rows.append((min(100.0, max(0.0, locked) + max(0.0, burned)), max(weight, 0.0)))
        if not rows:
            return None
        total_w = sum(w for _, w in rows)
        if total_w > 0:
            return sum(p * w for p, w in rows) / total_w
        return sum(p for p, _ in rows) / len(rows)

    async def _check(self, token_address: str, chain: str) -> SafetyVerdict:
        if chain != "solana":
            return SafetyVerdict.fail(self.name, f"unsupported chain {chain}")
        if not _SOL_RE.match(token_address):
            return SafetyVerdict.fail(self.name, "invalid Solana mint address")
        report = await self.fetcher.get_json(f"{self.base_url}/v1/tokens/{token_address}/report")
        if not isinstance(report, dict):
            return SafetyVerdict.fail(self.name, "malformed report")

        why: List[str] = []
        metrics: Dict[str, Any] = {}
        for label, camel, snake in (("mintAuthority", "mintAuthority", "mint_authority"),
                                    ("freezeAuthority", "freezeAuthority", "freeze_authority")):
            val = self._authority(report, camel, snake)
            if val is _MISSING:
                why.append(f"{label} missing from report")
            else:
                metrics[snake] = val          # None => renounced
                if val is not None:
                    why.append(f"{label} is set ({val})")

        score = _num(report.get("score"))
        metrics["score"] = score
        if score is None:
            why.append("score missing")
        elif score > self.max_score:
            why.append(f"risk score {score:g} > {self.max_score:g}")

        if report.get("rugged") is True:
            why.append("token flagged as rugged")

        risks = report.get("risks")
        if not isinstance(risks, list):
            why.append("risks missing")
        else:
            for r in risks:
                if not isinstance(r, dict):
                    why.append("malformed risk entry")
                    continue
                name, level = str(r.get("name", "")), str(r.get("level", "")).lower()
                if level in ("danger", "critical"):
                    why.append(f"risk '{name}' level={level}")
                elif "honeypot" in name.lower():
                    why.append(f"risk '{name}' mentions honeypot")

        lp = self._lp_secure_pct(report)
        metrics["lp_secure_pct"] = lp
        metrics["rugged"] = report.get("rugged")
        if isinstance(risks, list):
            metrics["risk_names"] = [str(r.get("name", "")) for r in risks if isinstance(r, dict)]
        if lp is None:
            why.append("LP lock/burn data missing")
        elif lp < self.min_lp:
            why.append(f"LP locked+burned {lp:.1f}% < {self.min_lp:g}%")

        return (SafetyVerdict.fail(self.name, *why, metrics=metrics) if why
                else SafetyVerdict.ok(self.name, metrics))


class GoPlusProvider(_HttpProvider):
    """EVM tokens via GET https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={addr}"""
    name = "goplus"
    FLAGS = ("is_honeypot", "cannot_sell_all", "trading_cooldown", "slippage_modifiable")

    def __init__(self, fetcher: JsonFetcher, max_tax: float = 0.10,
                 base_url: str = "https://api.gopluslabs.io",
                 headers: Optional[Mapping[str, str]] = None, **kw: Any) -> None:
        super().__init__(fetcher, **kw)
        self.max_tax, self.base_url, self.headers = max_tax, base_url.rstrip("/"), headers

    async def _check(self, token_address: str, chain: str) -> SafetyVerdict:
        chain_id = GOPLUS_CHAIN_IDS.get(chain)
        if chain_id is None:
            return SafetyVerdict.fail(self.name, f"unsupported chain {chain}")
        if not _EVM_RE.match(token_address):
            return SafetyVerdict.fail(self.name, "invalid EVM address")
        addr = token_address.lower()
        url = f"{self.base_url}/api/v1/token_security/{chain_id}?contract_addresses={addr}"
        data = await self.fetcher.get_json(url, self.headers)
        if not isinstance(data, dict) or _num(data.get("code")) != 1:
            code = data.get("code") if isinstance(data, dict) else None
            return SafetyVerdict.fail(self.name, f"API code {code!r} (rate limit or error)")
        result = data.get("result")
        entry = result.get(addr) if isinstance(result, dict) else None
        if not isinstance(entry, dict) or not entry:
            return SafetyVerdict.fail(self.name, "no security data for token")

        why: List[str] = []
        metrics: Dict[str, Any] = {f: str(entry.get(f, "")).strip() for f in self.FLAGS}
        if "owner_address" in entry:                       # best-effort optional GoPlus fields
            metrics["owner_address"] = str(entry.get("owner_address") or "")
        lph = entry.get("lp_holders")
        if isinstance(lph, list) and lph:
            metrics["lp_locked_pct"] = 100.0 * sum(
                (_num(h.get("percent")) or 0.0) for h in lph
                if isinstance(h, dict) and str(h.get("is_locked")) == "1")
        for f in self.FLAGS:
            v = str(entry.get(f, "")).strip()
            if v not in ("0", "1"):
                why.append(f"{f} missing/invalid")
            elif v == "1":
                why.append(f"{f} == 1")
        for t in ("buy_tax", "sell_tax"):
            v = _num(entry.get(t))
            metrics[t] = v
            if v is None:
                why.append(f"{t} missing/invalid")
            elif v > self.max_tax + 1e-9:
                why.append(f"{t} {v:.1%} > {self.max_tax:.0%}")
        return (SafetyVerdict.fail(self.name, *why, metrics=metrics) if why
                else SafetyVerdict.ok(self.name, metrics))


class CompositeSafetyProvider(SafetyDataProvider):
    """Routes Solana -> RugCheck, Ethereum/BSC/Base/Arbitrum -> GoPlus. Unknown chain => False."""

    def __init__(self, solana: Optional[_HttpProvider] = None, evm: Optional[_HttpProvider] = None,
                 fetcher: Optional[JsonFetcher] = None, **provider_kw: Any) -> None:
        self.fetcher = fetcher or AiohttpFetcher()
        self.solana = solana or RugCheckProvider(self.fetcher, **provider_kw)
        self.evm = evm or GoPlusProvider(self.fetcher, **provider_kw)

    def _route(self, chain: str) -> Optional[_HttpProvider]:
        norm = normalize_chain(chain)
        if norm == "solana":
            return self.solana
        return self.evm if norm in GOPLUS_CHAIN_IDS else None

    async def evaluate(self, token_address: str, chain: str) -> SafetyVerdict:
        p = self._route(chain)
        if p is None:
            return SafetyVerdict.fail("composite", f"unknown chain {chain!r}")
        return await p.evaluate(token_address, chain)

    async def is_safe(self, token_address: str, chain: str) -> bool:
        try:
            return (await self.evaluate(token_address, chain)).safe
        except Exception:  # noqa: BLE001
            log.exception("composite check crashed")
            return False

    async def evaluate_many(self, items: Sequence[Tuple[str, str]]) -> List[SafetyVerdict]:
        return list(await asyncio.gather(*(self.evaluate(a, c) for a, c in items)))

    async def aclose(self) -> None:
        await self.fetcher.aclose()

    async def __aenter__(self) -> "CompositeSafetyProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


# =============================================================================
# Test harness (offline mocks by default)
# =============================================================================
class MockFetcher(JsonFetcher):
    """Maps a URL substring to a payload, or an Exception to raise."""

    def __init__(self, routes: Mapping[str, Any]) -> None:
        self.routes, self.urls = dict(routes), []

    async def get_json(self, url: str, headers: Optional[Mapping[str, str]] = None) -> Any:
        self.urls.append(url)
        for frag, payload in self.routes.items():
            if frag in url:
                if isinstance(payload, Exception):
                    raise payload
                return copy.deepcopy(payload)
        raise FetchError(404, url)


def _mint(ch: str) -> str:
    return ch * 44  # syntactically valid base58 placeholder


def _evm(n: int) -> str:
    return "0x" + f"{n:040x}"


def _good_rugcheck() -> Dict[str, Any]:
    return {"score": 800, "rugged": False, "mintAuthority": None, "freezeAuthority": None,
            "risks": [{"name": "Mutable metadata", "level": "warn", "score": 100}],
            "markets": [{"lp": {"lpLockedPct": 95.0, "lpBurnedPct": 0, "baseUSD": 5e4, "quoteUSD": 5e4}}]}


def _good_goplus(addr: str) -> Dict[str, Any]:
    return {"code": 1, "message": "OK", "result": {addr.lower(): {
        "is_honeypot": "0", "cannot_sell_all": "0", "trading_cooldown": "0",
        "slippage_modifiable": "0", "buy_tax": "0.02", "sell_tax": "0.03"}}}


def _variant(base: Dict[str, Any], mutate) -> Dict[str, Any]:
    d = copy.deepcopy(base)
    mutate(d)
    return d


async def _run_mock_suite() -> int:
    routes: Dict[str, Any] = {}
    cases: List[Tuple[str, str, str, bool]] = []   # (label, address, chain, expected)

    def sol(label: str, ch: str, payload: Any, expect: bool, chain: str = "solana") -> None:
        routes[_mint(ch)] = payload
        cases.append((label, _mint(ch), chain, expect))

    g = _good_rugcheck()
    sol("SOL good", "A", g, True)
    sol("SOL mint authority set", "B", _variant(g, lambda d: d.update(mintAuthority="Auth111")), False)
    sol("SOL freeze authority set", "C", _variant(g, lambda d: d.update(freezeAuthority="Auth222")), False)
    sol("SOL freeze key absent", "D", _variant(g, lambda d: d.pop("freezeAuthority")), False)
    sol("SOL score 6000", "E", _variant(g, lambda d: d.update(score=6000)), False)
    sol("SOL LP 40%", "F", _variant(g, lambda d: d["markets"][0]["lp"].update(lpLockedPct=40.0)), False)
    sol("SOL danger risk", "G", _variant(g, lambda d: d["risks"].append(
        {"name": "Top holder owns 60%", "level": "danger", "score": 4000})), False)
    sol("SOL honeypot-named risk", "H", _variant(g, lambda d: d["risks"].append(
        {"name": "Possible Honeypot", "level": "warn", "score": 10})), False)
    sol("SOL score missing", "J", _variant(g, lambda d: d.pop("score")), False)
    sol("SOL HTTP 429", "K", FetchError(429, "slow down"), False)
    sol("SOL timeout", "L", FetchError(0, "timeout"), False)
    cases.append(("SOL invalid address", "not-a-mint", "solana", False))

    gp_addr = {n: _evm(n) for n in range(1, 16)}

    def evm(label: str, n: int, payload: Any, expect: bool, chain: str = "base") -> None:
        routes[gp_addr[n]] = payload
        cases.append((label, gp_addr[n], chain, expect))

    evm("EVM base good", 1, _good_goplus(gp_addr[1]), True)
    for n, (chain, cid) in enumerate((("ethereum", 1), ("bsc", 56), ("arbitrum", 42161)), start=2):
        evm(f"EVM {chain} good (id {cid})", n, _good_goplus(gp_addr[n]), True, chain)
    for n, f in enumerate(GoPlusProvider.FLAGS, start=5):
        evm(f"EVM {f}=1", n, _variant(_good_goplus(gp_addr[n]),
            lambda d, f=f, n=n: d["result"][gp_addr[n]].update({f: "1"})), False)
    evm("EVM sell_tax 25%", 9, _variant(_good_goplus(gp_addr[9]),
        lambda d: d["result"][gp_addr[9]].update(sell_tax="0.25")), False)
    evm("EVM buy_tax 11%", 10, _variant(_good_goplus(gp_addr[10]),
        lambda d: d["result"][gp_addr[10]].update(buy_tax="0.11")), False)
    evm("EVM buy_tax blank", 11, _variant(_good_goplus(gp_addr[11]),
        lambda d: d["result"][gp_addr[11]].update(buy_tax="")), False)
    evm("EVM empty result", 12, {"code": 1, "message": "OK", "result": {}}, False)
    evm("EVM API code 4029", 13, {"code": 4029, "message": "too many requests"}, False)
    evm("EVM timeout", 14, FetchError(0, "timeout"), False)
    cases.append(("Unknown chain polygon", gp_addr[1], "polygon", False))

    fetcher = MockFetcher(routes)
    comp = CompositeSafetyProvider(fetcher=fetcher, cache_ttl_s=0.0)
    verdicts = await comp.evaluate_many([(a, c) for _, a, c, _ in cases])
    bad = 0
    print(f"{'case':28s} {'expected':8s} {'got':6s} reasons")
    for (label, _, _, expect), v in zip(cases, verdicts):
        flag = "" if v.safe == expect else "   <-- MISMATCH"
        bad += v.safe != expect
        print(f"{label:28s} {str(expect):8s} {str(v.safe):6s} {'; '.join(v.reasons)[:70]}{flag}")
    ids = {"ethereum": "/1?", "bsc": "/56?", "base": "/8453?", "arbitrum": "/42161?"}
    for chain, frag in ids.items():
        if not any(frag in u for u in fetcher.urls):
            print(f"chain-id check failed for {chain}")
            bad += 1
    print(f"\n{'ALL OK' if bad == 0 else str(bad) + ' PROBLEM(S)'} ({len(cases)} cases)")
    print("default FailClosedProvider ->", await FailClosedProvider().is_safe(_mint("A"), "solana"))
    return 1 if bad else 0


async def _run_live() -> None:
    bonk = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    base_usdc = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    async with CompositeSafetyProvider() as comp:
        for addr, chain in ((bonk, "solana"), (base_usdc, "base")):
            v = await comp.evaluate(addr, chain)
            print(f"[live] {chain:7s} {addr[:10]}... safe={v.safe} {'; '.join(v.reasons)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also query the real APIs")
    args = ap.parse_args()
    code = asyncio.run(_run_mock_suite())
    if args.live:
        asyncio.run(_run_live())
    sys.exit(code)
''',
}


def _bootstrap_embedded() -> None:
    """Create each embedded module once per process (Streamlit reruns this script on every
    interaction; sys.modules persists, so we avoid re-exec and Numba recompilation)."""
    for name in ("memecoin_quant", "dexscreener_adapter", "safety_module"):   # dependency order
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__file__ = f"<embedded {name}>"
        sys.modules[name] = mod           # must exist before exec (dataclasses resolve annotations via it)
        try:
            exec(compile(_EMBEDDED[name], f"<embedded {name}>", "exec"), mod.__dict__)
        except Exception:
            sys.modules.pop(name, None)
            raise


_bootstrap_embedded()

# =============================================================================
# APPLICATION
# =============================================================================

import asyncio
import dataclasses
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import safety_module as sm  # noqa: E402
from dexscreener_adapter import (AiohttpTransport, DexScreenerClient,  # noqa: E402
                                 DexScreenerDiscoverySource, DexScreenerSourceConfig,
                                 FixtureTransport, PairSnapshot, Transport, _mk_pair, _norm,
                                 _same, best_pair)
from memecoin_quant import (CHAINS, AMMExecutionSim, Chain, Context, ContractSnapshot,  # noqa: E402
                            DataIngestion, DiscoveryEngine, DiscoverySource, EmaMomentumStrategy,
                            EventDrivenBacktester, EventStrategy, Order, PerformanceAnalytics,
                            PoolState, SafetyChecker, Side, TokenCandidate, TradeRecord,
                            ZERO_ADDRESSES, generate_mock_market)

# =============================================================================
# 0. Constants
# =============================================================================
GT_BASE = "https://api.geckoterminal.com/api/v2"          # free tier: 30 calls/min, no key
GT_NETWORKS = {"solana": "solana", "base": "base"}
# label -> (GeckoTerminal timeframe, aggregate, seconds per bar, pandas freq)
TIMEFRAMES: Dict[str, Tuple[str, int, int, str]] = {
    "1m": ("minute", 1, 60, "1min"), "5m": ("minute", 5, 300, "5min"),
    "15m": ("minute", 15, 900, "15min"), "1h": ("hour", 1, 3600, "1h"),
    "4h": ("hour", 4, 14400, "4h"), "1d": ("day", 1, 86400, "1D"),
}
MAX_TOKENS = 5            # keeps us inside GeckoTerminal's 30 calls/min
ZERO = {a.lower() for a in ZERO_ADDRESSES}


class DataError(Exception):
    """User-presentable data problem (bad response, too little history, ...)."""


# =============================================================================
# 1. THE BRIDGE: DexScreener candidates  ->  Safety Module  ->  scoring engine
# =============================================================================
@dataclass
class AuditLimits:
    max_score: float = 5000.0     # RugCheck raw score ceiling
    min_lp_pct: float = 80.0      # LP locked + burned floor
    max_tax_pct: float = 10.0     # GoPlus buy/sell tax ceiling


@dataclass
class AuditResult:
    chain: str
    address: str
    verdict: sm.SafetyVerdict
    snapshot: ContractSnapshot

    @property
    def safe(self) -> bool:
        return self.verdict.safe

    def check_rows(self, lim: AuditLimits) -> List[Dict[str, str]]:
        """Human-readable pass/fail table built from the RAW values the provider used."""
        m, rows = dict(self.verdict.metrics), []
        P, F, I = "✅ PASS", "❌ FAIL", "ℹ️ info"

        def add(check: str, value: str, thr: str, ok: Optional[bool]) -> None:
            rows.append({"Check": check, "Value": value, "Threshold": thr,
                         "Status": I if ok is None else (P if ok else F)})

        if not m:
            add("Provider response", "; ".join(self.verdict.reasons), "usable report", False)
            return rows
        if self.chain == "solana":
            for label, key in (("Mint authority", "mint_authority"), ("Freeze authority", "freeze_authority")):
                if key not in m:
                    add(label, "missing", "null", False)
                else:
                    add(label, "renounced (null)" if m[key] is None else f"ACTIVE: {m[key]}", "null", m[key] is None)
            lp = m.get("lp_secure_pct")
            add("LP locked + burned", "n/a" if lp is None else f"{lp:.1f}%", f">= {lim.min_lp_pct:g}%",
                lp is not None and lp >= lim.min_lp_pct)
            sc = m.get("score")
            add("RugCheck risk score", "n/a" if sc is None else f"{sc:g}", f"<= {lim.max_score:g}",
                sc is not None and sc <= lim.max_score)
            add("Flagged rugged", str(m.get("rugged")), "False", m.get("rugged") is not True)
            for r in m.get("risk_names", [])[:6]:
                add("Risk listed", r, "-", None)
        else:
            for f in sm.GoPlusProvider.FLAGS:
                add(f, str(m.get(f)), "0", m.get(f) == "0")
            for t in ("buy_tax", "sell_tax"):
                v = m.get(t)
                add(t.replace("_", " ").title(), "n/a" if v is None else f"{v:.1%}", f"<= {lim.max_tax_pct:g}%",
                    v is not None and v <= lim.max_tax_pct / 100 + 1e-9)
            if "owner_address" in m:
                own = m["owner_address"]
                add("Owner", "renounced" if (own == "" or own.lower() in ZERO) else own, "info", None)
            if "lp_locked_pct" in m:
                add("LP locked (GoPlus)", f"{m['lp_locked_pct']:.1f}%", "info (not enforced)", None)
        return rows


def snapshot_from_verdict(v: sm.SafetyVerdict, chain: Chain, address: str) -> ContractSnapshot:
    """
    Translate the safety module's raw metrics into the ContractSnapshot that the scoring
    engine's SafetyChecker consumes. Unknown stays UNKNOWN (never silently 'renounced'/'clean').
    """
    m = dict(v.metrics)
    if chain is Chain.SOLANA:
        return ContractSnapshot(
            address=address, chain=chain,
            mint_authority=m["mint_authority"] if "mint_authority" in m else "unknown",
            freeze_authority=m["freeze_authority"] if "freeze_authority" in m else "unknown",
            lp_locked_pct=float(m.get("lp_secure_pct") or 0.0))
    owner: Optional[str] = str(m.get("owner_address", "unknown"))
    if owner == "" or owner.lower() in ZERO:
        owner = None
    sell_ok = (m.get("is_honeypot") == "0" and m.get("cannot_sell_all") == "0") if "is_honeypot" in m else None
    return ContractSnapshot(
        address=address, chain=chain, owner=owner,
        buy_tax_bps=float(m.get("buy_tax") or 0.0) * 1e4, sell_tax_bps=float(m.get("sell_tax") or 0.0) * 1e4,
        lp_locked_pct=float(m.get("lp_locked_pct") or 0.0), sell_simulation_ok=sell_ok)


class SafetyBridge:
    """Runs the fail-closed CompositeSafetyProvider and packages the result for the rest of the app."""

    def __init__(self, composite: sm.CompositeSafetyProvider) -> None:
        self.composite = composite

    async def audit(self, chain: Chain, address: str) -> AuditResult:
        try:
            v = await self.composite.evaluate(address, chain.value)
        except Exception as exc:  # noqa: BLE001 -- the bridge itself must also fail closed
            v = sm.SafetyVerdict.fail("bridge", f"audit crashed: {type(exc).__name__}")
        return AuditResult(chain.value, address, v, snapshot_from_verdict(v, chain, address))

    async def audit_many(self, items: Sequence[Tuple[Chain, str]]) -> List[AuditResult]:
        return list(await asyncio.gather(*(self.audit(c, a) for c, a in items)))


class GatedDiscoverySource(DiscoverySource):
    """Wraps DexScreenerDiscoverySource: drops candidates the safety module rejects and attaches
    audited snapshots to the survivors, so DiscoveryEngine scores real contract data."""
    name = "dexscreener+safety"

    def __init__(self, inner: DexScreenerDiscoverySource, bridge: SafetyBridge) -> None:
        self.inner, self.bridge = inner, bridge
        self.rejected: List[Dict[str, Any]] = []

    async def fetch_candidates(self) -> List[TokenCandidate]:
        self.rejected = []
        cands = await self.inner.fetch_candidates()
        audits = await self.bridge.audit_many([(c.chain, c.snapshot.address) for c in cands])
        kept: List[TokenCandidate] = []
        for c, a in zip(cands, audits):
            if a.safe:
                c.snapshot = a.snapshot
                kept.append(c)
            else:
                self.rejected.append({"token": c.token, "provider": a.verdict.provider,
                                      "reasons": "; ".join(a.verdict.reasons)})
        return kept


# =============================================================================
# 2. Raw-response recording (feeds the "Execution Log" expander)
# =============================================================================
def _record(log: List[dict], source: str, url: str, t0: float, ok: bool, payload: Any) -> None:
    log.append({"source": source, "url": url, "ok": ok, "ms": round((time.time() - t0) * 1000),
                "at": time.strftime("%H:%M:%S"), "response" if ok else "error": payload})


class RecordingTransport(Transport):
    """DexScreener transport wrapper that logs every call."""

    def __init__(self, inner: Transport, log: List[dict]) -> None:
        self.inner, self.log = inner, log

    async def get_json(self, path: str, params: Optional[Mapping[str, str]] = None) -> Any:
        t0 = time.time()
        try:
            data = await self.inner.get_json(path, params)
        except Exception as exc:
            _record(self.log, "dexscreener", path, t0, False, str(exc))
            raise
        _record(self.log, "dexscreener", path, t0, True, data)
        return data

    async def close(self) -> None:
        await self.inner.close()


class RecordingFetcher(sm.JsonFetcher):
    """Safety / GeckoTerminal fetcher wrapper that logs every call."""

    def __init__(self, inner: sm.JsonFetcher, log: List[dict], source: str) -> None:
        self.inner, self.log, self.source = inner, log, source

    async def get_json(self, url: str, headers: Optional[Mapping[str, str]] = None) -> Any:
        t0 = time.time()
        try:
            data = await self.inner.get_json(url, headers)
        except Exception as exc:
            _record(self.log, self.source, url, t0, False, str(exc))
            raise
        _record(self.log, self.source, url, t0, True, data)
        return data

    async def aclose(self) -> None:
        await self.inner.aclose()


# =============================================================================
# 3. Offline demo world (lets the app run with zero network access)
# =============================================================================
class DemoGeckoFetcher(sm.JsonFetcher):
    """Serves GeckoTerminal-shaped OHLCV JSON from synthetic data, so the real parser is exercised."""

    def __init__(self, pools: Mapping[str, pd.DataFrame]) -> None:
        self.pools = dict(pools)

    async def get_json(self, url: str, headers: Optional[Mapping[str, str]] = None) -> Any:
        path = urlparse(url).path
        pool = path.split("/pools/")[1].split("/ohlcv/")[0]
        df = self.pools.get(pool)
        if df is None:
            raise sm.FetchError(404, f"unknown demo pool {pool}")
        limit = int(parse_qs(urlparse(url).query).get("limit", ["100"])[0])
        d = df.tail(limit)
        rows = [[int(ts.tz_localize("UTC").timestamp()), r.open, r.high, r.low, r.close, r.volume]
                for ts, r in zip(d.index, d.itertuples())][::-1]          # API returns newest first
        return {"data": {"id": "demo", "attributes": {"ohlcv_list": rows}}}


@dataclass
class DemoWorld:
    addresses: List[str]
    dex: FixtureTransport
    safety: sm.MockFetcher
    gecko: DemoGeckoFetcher


def demo_world(chain: str, n: int = 400, tf: str = "15m") -> DemoWorld:
    """Two synthetic tokens per chain: one that passes the audit, one that fails it."""
    freq = TIMEFRAMES[tf][3]
    if chain == "solana":
        safe, risky = ("SafeDemo" + "1" * 36)[:44], ("RiskDemo" + "2" * 36)[:44]
        quote_sym = "SOL"
    else:
        safe, risky = "0x" + "5afe" * 10, "0x" + "bad0" * 10
        quote_sym = "WETH"
    start = (pd.Timestamp.now().floor(freq) - n * pd.Timedelta(freq)).isoformat()
    specs = {safe: ("SAFE", 11, None), risky: ("RISKY", 22, int(n * 0.72))}
    frames: Dict[str, pd.DataFrame] = {}
    pairs: Dict[str, Dict[str, Any]] = {}
    for addr, (sym, seed, rug) in specs.items():
        m = generate_mock_market(n_bars=n, seed=seed, start=start, freq=freq, rug_at=None,
                                 delay_bars=max(10, n // 6), init_liq_quote=200.0)
        rng = np.random.default_rng(seed)
        c = m["close"].iloc[0] * (m["close"] / m["close"].iloc[0]) ** 0.22   # tame the ~1e6x mock pump to ~20x
        if rug is not None:
            c.iloc[rug:] *= 0.03                                              # simulated rug: -97% in one bar
        o = c.shift(1).fillna(c.iloc[0])
        wick = np.abs(rng.normal(0, 0.004, n))
        ohlc = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + wick),
                             "low": np.minimum(o, c) * (1 - wick), "close": c,
                             "volume": m["volume"] * 150.0}, index=m.index)
        frames[f"pair_{sym}"] = ohlc
        pairs[addr] = _mk_pair(chain, addr, sym, quote_sym, price=float(c.iloc[-1]),
                               liq_quote=float(m["reserve_quote"].iloc[-1]), age_h=min(n * TIMEFRAMES[tf][2] / 3600, 48.0))

    def tokens_route(path: str, params: Any) -> List[Dict[str, Any]]:
        wanted = path.rsplit("/", 1)[-1].split(",")
        return [pairs[a] for a in wanted if a in pairs]

    dex = FixtureTransport({
        "/token-profiles/latest/v1": [{"chainId": chain, "tokenAddress": a} for a in specs],
        "/token-boosts/latest/v1": [],
        "/tokens/v1/": tokens_route})
    routes: Dict[str, Any] = {}
    if chain == "solana":
        routes[safe] = sm._good_rugcheck()
        routes[risky] = sm._variant(sm._good_rugcheck(), lambda d: (
            d.update(mintAuthority="DemoAuth1111"), d["markets"][0]["lp"].update(lpLockedPct=30.0)))
    else:
        routes[safe.lower()] = sm._variant(sm._good_goplus(safe), lambda d: d["result"][safe.lower()].update(
            owner_address="", is_open_source="1", lp_holders=[{"address": "0xlocker", "is_locked": "1", "percent": "0.95"}]))
        routes[risky.lower()] = sm._variant(sm._good_goplus(risky), lambda d: d["result"][risky.lower()].update(
            is_honeypot="1", sell_tax="0.30"))
    return DemoWorld(list(specs), dex, sm.MockFetcher(routes), DemoGeckoFetcher(frames))


# =============================================================================
# 4. Data pipeline: DexScreener pairs + safety audit + GeckoTerminal candles
# =============================================================================
@dataclass
class FetchConfig:
    chain: str
    addresses: List[str]
    demo: bool
    tf: str
    limit: int
    limits: AuditLimits
    require_native_quote: bool = True


@dataclass
class TokenData:
    address: str
    pair: Optional[PairSnapshot] = None
    audit: Optional[AuditResult] = None
    ohlcv: Optional[pd.DataFrame] = None
    error: Optional[str] = None


@dataclass
class PipelineData:
    cfg: FetchConfig
    tokens: Dict[str, TokenData]
    log: List[dict]
    notes: List[str]
    fetched_at: str


def valid_address(chain: str, addr: str) -> bool:
    return bool((sm._SOL_RE if chain == "solana" else sm._EVM_RE).match(addr))


async def fetch_ohlcv(fetcher: sm.JsonFetcher, network: str, pool: str, tf: str, limit: int,
                      token_side: str) -> pd.DataFrame:
    """GeckoTerminal OHLCV, priced in the pool's counter asset (SOL/ETH) via currency=token."""
    unit, agg, _, _ = TIMEFRAMES[tf]
    url = (f"{GT_BASE}/networks/{network}/pools/{quote(pool, safe='')}/ohlcv/{unit}"
           f"?aggregate={agg}&limit={int(limit)}&currency=token&token={token_side}")
    try:
        payload = await fetcher.get_json(url, {"Accept": "application/json"})
    except sm.FetchError as exc:
        raise DataError(f"GeckoTerminal request failed: {exc}") from exc
    try:
        rows = payload["data"]["attributes"]["ohlcv_list"]
    except (KeyError, TypeError) as exc:
        raise DataError("Unexpected GeckoTerminal response shape") from exc
    if not rows:
        raise DataError("GeckoTerminal returned no candles for this pool")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).astype(float)
    df.index = pd.to_datetime(df.pop("ts"), unit="s")
    df = df[~df.index.duplicated()].sort_index().dropna()
    df = df[df["close"] > 0]
    if len(df) < 30:
        raise DataError(f"Only {len(df)} candles available (need >= 30); try a shorter timeframe")
    return df


async def run_fetch(cfg: FetchConfig) -> PipelineData:
    log: List[dict] = []
    notes: List[str] = []
    chain = Chain(cfg.chain)
    tokens = {a: TokenData(a) for a in cfg.addresses}
    valid = []
    for a, td in tokens.items():
        if valid_address(cfg.chain, a):
            valid.append(a)
        else:
            td.error = f"Not a valid {cfg.chain} address"

    # ---- wire transports (live vs offline demo)
    if cfg.demo:
        world = demo_world(cfg.chain, cfg.limit, cfg.tf)
        dex_t: Transport = RecordingTransport(world.dex, log)
        safety_f = RecordingFetcher(world.safety, log, "safety")
        gecko_f = RecordingFetcher(world.gecko, log, "geckoterminal")
    else:
        dex_t = RecordingTransport(AiohttpTransport(), log)
        safety_f = RecordingFetcher(sm.AiohttpFetcher(timeout_s=6.0), log, "safety")
        gecko_f = RecordingFetcher(sm.AiohttpFetcher(timeout_s=10.0), log, "geckoterminal")

    client = DexScreenerClient(dex_t)
    lim = cfg.limits
    composite = sm.CompositeSafetyProvider(
        solana=sm.RugCheckProvider(safety_f, max_score=lim.max_score, min_lp_secure_pct=lim.min_lp_pct,
                                   cache_ttl_s=0.0),
        evm=sm.GoPlusProvider(safety_f, max_tax=lim.max_tax_pct / 100.0, cache_ttl_s=0.0),
        fetcher=safety_f)
    bridge = SafetyBridge(composite)
    try:
        if not valid:
            return PipelineData(cfg, tokens, log, notes, time.strftime("%H:%M:%S"))

        # ---- DexScreener pairs + security audits run concurrently
        async def get_pairs() -> List[PairSnapshot]:
            try:
                return await client.tokens(cfg.chain, valid)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"DexScreener request failed: {exc}")
                return []

        pairs, audits = await asyncio.gather(get_pairs(), bridge.audit_many([(chain, a) for a in valid]))
        dex_failed = any(e["source"] == "dexscreener" and not e["ok"] for e in log)
        index: Dict[str, List[PairSnapshot]] = {}
        for p in pairs:
            index.setdefault(_norm(p.base_addr), []).append(p)
            index.setdefault(_norm(p.quote_addr), []).append(p)
        for a, au in zip(valid, audits):
            td = tokens[a]
            td.audit = au
            td.pair = best_pair(index.get(_norm(a), []), a, require_native_quote=cfg.require_native_quote)
            if td.pair is None:
                td.error = ("DexScreener request failed (see raw log)" if dex_failed else
                            "No modelable constant-product pair found (CLMM / non-native quote / "
                            "inconsistent reserves / unknown token)")

        # ---- candles (sequential to respect GeckoTerminal's 30/min)
        for a in valid:
            td = tokens[a]
            if td.pair is None:
                continue
            side = "base" if _same(a, td.pair.base_addr) else "quote"
            try:
                td.ohlcv = await fetch_ohlcv(gecko_f, GT_NETWORKS[cfg.chain], td.pair.pair_address,
                                             cfg.tf, cfg.limit, side)
            except DataError as exc:
                td.error = str(exc)
        return PipelineData(cfg, tokens, log, notes, time.strftime("%H:%M:%S"))
    finally:
        for closer in (composite.aclose(), client.close(), gecko_f.aclose()):
            try:
                await closer
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# 5. Backtest: audit-gated signal replay through the AMM execution simulator
# =============================================================================
@dataclass
class BtConfig:
    capital_usd: float = 1000.0
    position_pct: float = 20.0
    stop_loss_pct: float = 30.0       # 0 = off
    take_profit_pct: float = 0.0      # 0 = off
    trail_pct: float = 20.0           # 0 = off
    max_impact_pct: float = 5.0       # slippage tolerance: reject fills whose curve impact exceeds this
    ema_fast: int = 5
    ema_slow: int = 20
    buy_tax_pct: float = 0.0
    sell_tax_pct: float = 0.0
    marketing_pct: float = 0.0
    use_detected_taxes: bool = True
    pool_fee_bps: float = 25.0
    congestion: float = 0.0
    liquidity_haircut_pct: float = 0.0
    gate_override: bool = False


class SignalReplayStrategy(EventStrategy):
    """EMA-cross signals + hard stop / take-profit / trailing stop. Buys only when the gate is open.
    Orders are filled on the NEXT bar by the engine (no lookahead)."""

    def __init__(self, token: str, entries: Mapping[pd.Timestamp, bool], exits: Mapping[pd.Timestamp, bool],
                 cfg: BtConfig, gate_open: bool, gas_buffer: float) -> None:
        self.token, self.entries, self.exits = token, entries, exits
        self.cfg, self.gate_open, self.gas_buffer = cfg, gate_open, gas_buffer

    def on_bar(self, ctx: Context) -> List[Order]:
        bar = ctx.bars.get(self.token)
        if bar is None:
            return []
        c, pos = self.cfg, ctx.portfolio.positions.get(self.token)
        if pos is not None:
            r = bar.close / pos.entry_ref - 1.0
            if self.exits.get(ctx.ts):
                return [Order(self.token, Side.SELL, tag="signal-exit")]
            if c.stop_loss_pct > 0 and r <= -c.stop_loss_pct / 100:
                return [Order(self.token, Side.SELL, tag="stop-loss")]
            if c.take_profit_pct > 0 and r >= c.take_profit_pct / 100:
                return [Order(self.token, Side.SELL, tag="take-profit")]
            if c.trail_pct > 0 and bar.close <= pos.peak * (1 - c.trail_pct / 100):
                return [Order(self.token, Side.SELL, tag="trailing-stop")]
            return []
        if self.gate_open and self.entries.get(ctx.ts):
            size = min(ctx.portfolio.cash * c.position_pct / 100, ctx.portfolio.cash - self.gas_buffer)
            if size > 0:
                return [Order(self.token, Side.BUY, quote_amount=size, tag="ema-cross")]
        return []


class LoggingBacktester(EventDrivenBacktester):
    """Same engine, but records every attempted fill (price, slippage, impact, gas, reason)."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.exec_log: List[Dict[str, Any]] = []

    def _execute(self, pf, o, ts, bars) -> None:  # type: ignore[no-untyped-def]
        bar = bars.get(o.token)
        before_in = o.token in pf.positions
        before_tokens = pf.positions[o.token].tokens if before_in else 0.0
        fill = None
        if bar is not None:       # dry-run the same fill purely for logging
            pool = PoolState(bar.reserve_token, bar.reserve_quote, self.sim.cfg.pool_fee_bps)
            if o.side is Side.BUY:
                fill = self.sim.buy(pool, o.quote_amount, bar.congestion)
            elif before_in:
                fill = self.sim.sell(pool, before_tokens * min(max(o.fraction, 0.0), 1.0), bar.congestion)
        super()._execute(pf, o, ts, bars)
        after_in = o.token in pf.positions
        after_tokens = pf.positions[o.token].tokens if after_in else 0.0
        executed = (after_in and not before_in) if o.side is Side.BUY else (after_tokens < before_tokens)
        reason = "" if executed else (fill.reason if fill is not None and not fill.success else "cash/position rule")
        self.exec_log.append({
            "ts": ts, "side": o.side.value, "tag": o.tag, "executed": executed, "reason": reason,
            "tokens": getattr(fill, "tokens", 0.0), "avg_price": getattr(fill, "avg_price", 0.0),
            "mid_price": getattr(fill, "mid_before", 0.0), "slippage_bps": getattr(fill, "slippage_bps", 0.0),
            "impact_bps": getattr(fill, "price_impact_bps", 0.0), "gas": getattr(fill, "gas", 0.0),
            "quote_spent_or_received": (getattr(fill, "total_cost", 0.0) if o.side is Side.BUY
                                        else getattr(fill, "net_quote", 0.0))})


@dataclass
class BtOutput:
    df: pd.DataFrame
    equity_usd: pd.Series
    trades: List[TradeRecord]
    exec_log: List[Dict[str, Any]]
    metrics: Dict[str, float]
    gate_open: bool
    blocked: bool
    quote_usd: float
    taxes_used: Tuple[float, float]
    ema: Tuple[np.ndarray, np.ndarray]
    notes: List[str]


def run_backtest(td: TokenData, chain: str, cfg: BtConfig, tf: str) -> BtOutput:
    assert td.pair is not None and td.ohlcv is not None and td.audit is not None
    notes: List[str] = []
    pair = td.pair
    o = pair.orient(td.address)
    if o is None:
        raise DataError("Pair has no usable reserves")
    # USD per quote asset (for capital conversion); falls back to 1:1 and says so
    if pair.price_usd and pair.price_native > 0:
        quote_usd = pair.price_usd / pair.price_native if _same(td.address, pair.base_addr) else pair.price_usd
    else:
        quote_usd = 1.0
        notes.append("No USD price from DexScreener: capital is treated as quote-asset units.")
    capital_q = cfg.capital_usd / quote_usd

    df = td.ohlcv[["open", "high", "low", "close", "volume"]].copy()
    # Reserves aren't available historically. Assume constant k (no LP adds/removes), so a price
    # path driven purely by swaps implies Q_t = Q_now * sqrt(P_t / P_now). Haircut scales both reserves.
    df["reserve_quote"] = (o.reserve_quote * (1 - cfg.liquidity_haircut_pct / 100)
                           * np.sqrt(df["close"] / o.price))
    df["social_mentions"] = 0.0
    df["congestion"] = cfg.congestion
    df = DataIngestion.validate(df)
    if len(df) < max(cfg.ema_slow * 2, 30):
        raise DataError("Not enough candles for the chosen EMA lengths")

    entries, exits = EmaMomentumStrategy(cfg.ema_fast, cfg.ema_slow).generate(df)
    ema = (_ema(df["close"].to_numpy(), cfg.ema_fast), _ema(df["close"].to_numpy(), cfg.ema_slow))

    # taxes: detected from the audit (EVM/GoPlus) unless the user overrides them
    m = td.audit.verdict.metrics
    buy_t, sell_t = cfg.buy_tax_pct, cfg.sell_tax_pct
    if cfg.use_detected_taxes and m.get("buy_tax") is not None and m.get("sell_tax") is not None:
        buy_t, sell_t = float(m["buy_tax"]) * 100, float(m["sell_tax"]) * 100
        notes.append(f"Using audited taxes: buy {buy_t:.1f}% / sell {sell_t:.1f}%.")
    chain_cfg = dataclasses.replace(CHAINS[Chain(chain)], pool_fee_bps=cfg.pool_fee_bps)
    sim = AMMExecutionSim(chain_cfg, buy_tax_bps=buy_t * 100, sell_tax_bps=sell_t * 100,
                          marketing_fee_bps=cfg.marketing_pct * 100, max_price_impact_bps=cfg.max_impact_pct * 100)

    gate_open = td.audit.safe or cfg.gate_override
    sym = pair.base_symbol if _same(td.address, pair.base_addr) else pair.quote_symbol
    strat = SignalReplayStrategy(sym, dict(zip(df.index, entries)), dict(zip(df.index, exits)), cfg,
                                 gate_open, sim.gas_cost(cfg.congestion) * 2)
    bars_per_year = 365.25 * 86400 / TIMEFRAMES[tf][2]
    bt = LoggingBacktester(sim, init_cash=capital_q, bars_per_year=bars_per_year)
    res = bt.run(DataIngestion.to_event_feed({sym: df}), strat)

    metrics = dict(res.metrics)
    fills = [e for e in bt.exec_log if e["executed"]]
    metrics["n_fills"] = float(len(fills))
    return BtOutput(df, res.equity * quote_usd, res.trades, bt.exec_log, metrics, gate_open,
                    blocked=not gate_open, quote_usd=quote_usd, taxes_used=(buy_t, sell_t), ema=ema, notes=notes)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    from memecoin_quant import ema_nb
    return ema_nb(np.asarray(x, dtype=np.float64), int(span))


# =============================================================================
# 6. Plotly chart: candles + EMAs, volume overlay, fills, equity curve
# =============================================================================
def build_chart(out: BtOutput, title: str) -> go.Figure:
    df = out.df
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.60, 0.17, 0.23],
                        subplot_titles=(title, "Volume (USD)", "Equity (USD)"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name="Price (quote)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=out.ema[0], mode="lines", name="EMA fast",
                             line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=out.ema[1], mode="lines", name="EMA slow",
                             line=dict(width=1)), row=1, col=1)
    up = (df["close"] >= df["open"]).to_numpy()
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color=np.where(up, "#26a69a", "#ef5350")), row=2, col=1)
    for side, symbol, color in (("buy", "triangle-up", "#16a34a"), ("sell", "triangle-down", "#dc2626")):
        pts = [e for e in out.exec_log if e["executed"] and e["side"] == side]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p["ts"] for p in pts], y=[p["avg_price"] for p in pts], mode="markers",
                name=f"{side.title()} fill", marker=dict(symbol=symbol, size=13, color=color,
                                                          line=dict(width=1, color="white")),
                text=[f"{p['tag']}<br>slippage {p['slippage_bps']:.0f} bps<br>impact {p['impact_bps']:.0f} bps"
                      for p in pts], hoverinfo="text+x+y"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out.equity_usd.index, y=out.equity_usd.values, mode="lines",
                             name="Equity (marked through the AMM curve)"), row=3, col=1)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# =============================================================================
# 7. Discovery tab runner (exercises the bridge end to end)
# =============================================================================
async def run_discovery(chain: str, demo: bool, limits: AuditLimits, min_liq: float,
                        max_age_h: float, max_cands: int, tf: str) -> Dict[str, Any]:
    log: List[dict] = []
    if demo:
        world = demo_world(chain, 300, tf)
        dex_t: Transport = RecordingTransport(world.dex, log)
        safety_f = RecordingFetcher(world.safety, log, "safety")
    else:
        dex_t = RecordingTransport(AiohttpTransport(), log)
        safety_f = RecordingFetcher(sm.AiohttpFetcher(timeout_s=6.0), log, "safety")
    client = DexScreenerClient(dex_t)
    composite = sm.CompositeSafetyProvider(
        solana=sm.RugCheckProvider(safety_f, max_score=limits.max_score, min_lp_secure_pct=limits.min_lp_pct),
        evm=sm.GoPlusProvider(safety_f, max_tax=limits.max_tax_pct / 100.0), fetcher=safety_f)
    inner = DexScreenerDiscoverySource(client, cfg=DexScreenerSourceConfig(
        chains={chain}, min_liquidity_usd=min_liq, max_pair_age_hours=max_age_h, max_candidates=max_cands))
    gated = GatedDiscoverySource(inner, SafetyBridge(composite))
    try:
        sugs = await DiscoveryEngine([gated], safety=SafetyChecker(max_risk=60.0)).poll_once()
    finally:
        await composite.aclose()
        await client.close()
    return {"suggestions": sugs, "rejected": gated.rejected, "log": log}


# =============================================================================
# 8. Streamlit UI
# =============================================================================
def _fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None or x != x else f"{x * 100:+.1f}%"


def _trim(obj: Any, n: int = 40) -> Any:
    """Keep the raw-JSON viewer responsive: truncate long lists."""
    if isinstance(obj, list):
        return [_trim(i, n) for i in obj[:n]] + ([f"... {len(obj) - n} more"] if len(obj) > n else [])
    if isinstance(obj, dict):
        return {k: _trim(v, n) for k, v in obj.items()}
    return obj


def sidebar() -> Dict[str, Any]:
    sb = st.sidebar
    sb.header("⚙️ Controls")
    mode = sb.radio("Data mode", ["Live APIs", "Offline demo"], help="Demo needs no network.")
    demo = mode == "Offline demo"
    chain_label = sb.selectbox("Chain", ["Solana", "Base"])
    chain = chain_label.lower()
    if demo:
        d = demo_world(chain, 60, "1h").addresses
        default_addrs = "\n".join(d)
    else:
        default_addrs = ("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263" if chain == "solana"
                         else "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed")
    addr_text = sb.text_area(f"Token addresses (one per line, max {MAX_TOKENS})", default_addrs,
                             key=f"addrs_{demo}_{chain}", height=110,
                             help="Examples only. Verify addresses yourself.")
    tf = sb.selectbox("Candle timeframe", list(TIMEFRAMES), index=2)
    limit = sb.slider("Candles", 100, 1000, 400, 50)

    sb.subheader("Capital & execution")
    bt = BtConfig()
    bt.capital_usd = sb.number_input("Initial capital (USD)", 10.0, 1e7, 1000.0, 100.0)
    bt.position_pct = sb.slider("Position size (% of cash)", 1, 100, 20)
    bt.max_impact_pct = sb.number_input("Slippage tolerance: max price impact (%)", 0.1, 90.0, 5.0, 0.5,
                                        help="Fills whose constant-product impact exceeds this are rejected.")
    bt.pool_fee_bps = sb.number_input("Pool LP fee (bps)", 0.0, 500.0, float(CHAINS[Chain(chain)].pool_fee_bps), 5.0,
                                      key=f"fee_{chain}")
    bt.congestion = sb.slider("Network congestion (gas)", 0.0, 1.0, 0.2, 0.05)
    bt.liquidity_haircut_pct = sb.slider("Liquidity haircut vs today (%)", 0, 90, 0,
                                         help="Stress test: assume past pool depth was smaller than today's.")
    bt.use_detected_taxes = sb.checkbox("Use taxes detected by the audit (EVM)", True)
    bt.buy_tax_pct = sb.number_input("Manual buy tax (%)", 0.0, 50.0, 0.0, 0.5)
    bt.sell_tax_pct = sb.number_input("Manual sell tax (%)", 0.0, 50.0, 0.0, 0.5)
    bt.marketing_pct = sb.number_input("Marketing fee (%)", 0.0, 10.0, 0.0, 0.25)

    sb.subheader("Strategy & risk limits")
    bt.ema_fast = sb.slider("EMA fast", 2, 50, 5)
    bt.ema_slow = sb.slider("EMA slow", 5, 200, 20)
    bt.stop_loss_pct = sb.number_input("Stop loss (%, 0 = off)", 0.0, 99.0, 30.0, 1.0)
    bt.take_profit_pct = sb.number_input("Take profit (%, 0 = off)", 0.0, 10000.0, 0.0, 10.0)
    bt.trail_pct = sb.number_input("Trailing stop (%, 0 = off)", 0.0, 99.0, 20.0, 1.0)

    sb.subheader("Security limits (applied on Run)")
    limits = AuditLimits(
        max_score=sb.number_input("Max RugCheck score", 0.0, 100000.0, 5000.0, 250.0),
        min_lp_pct=sb.number_input("Min LP locked+burned (%)", 0.0, 100.0, 80.0, 5.0),
        max_tax_pct=sb.number_input("Max buy/sell tax (%)", 0.0, 50.0, 10.0, 1.0))
    native_only = sb.checkbox("Require native-asset quote (SOL/ETH)", True,
                              help="The simulator's gas model is in native units.")
    bt.gate_override = sb.checkbox("⚠️ Override security gate (research only)", False)
    run = sb.button("▶ Run analysis", type="primary", use_container_width=True)
    addrs = list(dict.fromkeys(a.strip() for a in addr_text.splitlines() if a.strip()))
    return dict(demo=demo, chain=chain, addrs=addrs, tf=tf, limit=limit, bt=bt, limits=limits,
                native_only=native_only, run=run)


def render_security_tab(data: PipelineData, selected: Optional[str]) -> None:
    st.caption("Fail-closed: any timeout, rate limit, missing field or unknown chain counts as FAIL. "
               "Thresholds were applied when you pressed Run.")
    rows = []
    for a, td in data.tokens.items():
        if td.audit is None:
            rows.append({"Token": a[:10] + "…", "Status": "⛔ NOT AUDITED", "Provider": "-", "Why": td.error or ""})
        else:
            v = td.audit.verdict
            rows.append({"Token": a[:10] + "…", "Status": "✅ PASS" if v.safe else "❌ FAIL",
                         "Provider": v.provider, "Why": "; ".join(v.reasons) or "all checks passed"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    td = data.tokens.get(selected) if selected else None
    if td is not None and td.audit is not None:
        st.subheader(f"Checks for {selected[:12]}…")
        st.dataframe(pd.DataFrame(td.audit.check_rows(data.cfg.limits)), use_container_width=True, hide_index=True)
        with st.expander("Second-layer heuristic score (memecoin_quant.SafetyChecker)"):
            rep = SafetyChecker().check(td.audit.snapshot, None)
            st.write(f"Risk score **{rep.risk_score:.0f}/100** (pass < 50, no critical flags) → "
                     f"{'pass' if rep.passed else 'fail'}")
            st.write([f"{f.code} ({f.weight:g}): {f.message}" for f in rep.flags] or "no flags")
            st.caption("Source-code flags show 'UNVERIFIED_SOURCE' because neither API supplies contract source.")


def render_discovery_tab(ui: Dict[str, Any]) -> None:
    st.caption("Runs DexScreener → safety bridge → SIR/late-buyer scoring. Unsafe candidates are dropped before scoring.")
    c1, c2, c3 = st.columns(3)
    min_liq = c1.number_input("Min liquidity (USD)", 1000.0, 1e7, 10_000.0, 1000.0)
    max_age = c2.number_input("Max pair age (hours)", 1.0, 720.0, 72.0, 6.0)
    max_c = c3.number_input("Max candidates", 1, 30, 10, help="Each candidate costs 1-3 safety API calls.")
    if st.button("🔎 Scan latest DexScreener profiles/boosts"):
        with st.spinner("Scanning and auditing…"):
            try:
                st.session_state["disc"] = asyncio.run(run_discovery(
                    ui["chain"], ui["demo"], ui["limits"], min_liq, max_age, int(max_c), ui["tf"]))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Discovery failed: {exc}")
    d = st.session_state.get("disc")
    if d:
        sugs = d["suggestions"]
        st.dataframe(pd.DataFrame([{
            "Token": s.token, "Action": s.action, "Score": round(s.score, 3),
            "Internal risk": round(s.safety.risk_score), "Hrs since peak": round(s.components["hours_since_peak"], 1),
            "Late-buyer x": round(s.components["late_multiplier"], 2),
            "Liquidity USD": round(s.components["liquidity_usd"])} for s in sugs]) if sugs
            else pd.DataFrame({"result": ["no candidates survived the filters + safety gate"]}),
            use_container_width=True, hide_index=True)
        if d["rejected"]:
            st.markdown("**Rejected by the safety gate**")
            st.dataframe(pd.DataFrame(d["rejected"]), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Memecoin Backtester + Security Audit", layout="wide", page_icon="🧪")
    st.title("🧪 Memecoin Backtester + Security Audit")
    st.caption("Simulation only — not financial advice. Constant-product (x·y=k) execution with gas, taxes, "
               "fees and price-impact limits. Buys are blocked unless the token passes the audit.")
    ui = sidebar()

    # ---- fetch on demand; everything after this recomputes live from the stored data
    if ui["run"]:
        if not ui["addrs"]:
            st.warning("Enter at least one token address.")
        else:
            addrs = ui["addrs"][:MAX_TOKENS]
            if len(ui["addrs"]) > MAX_TOKENS:
                st.warning(f"Only the first {MAX_TOKENS} addresses are used (API rate limits).")
            cfg = FetchConfig(ui["chain"], addrs, ui["demo"], ui["tf"], ui["limit"], ui["limits"], ui["native_only"])
            with st.spinner("Fetching pairs, running security audits, loading candles…"):
                try:
                    st.session_state["data"] = asyncio.run(run_fetch(cfg))
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("data", None)
                    st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")

    data: Optional[PipelineData] = st.session_state.get("data")
    tab_bt, tab_sec, tab_disc = st.tabs(["📈 Backtest", "🛡️ Security Audit", "🔎 Discovery (bridge)"])
    with tab_disc:
        render_discovery_tab(ui)
    if data is None:
        with tab_bt:
            st.info("Set parameters in the sidebar and press **Run analysis**. "
                    "Pick *Offline demo* to try it without network access.")
        return

    for n in data.notes:
        st.warning(n)
    errs = {a: t.error for a, t in data.tokens.items() if t.error}
    if errs:
        with st.expander(f"⚠️ {len(errs)} token(s) had problems", expanded=not any(
                t.ohlcv is not None for t in data.tokens.values())):
            for a, e in errs.items():
                st.write(f"`{a[:14]}…` — {e}")
    usable = [a for a, t in data.tokens.items() if t.ohlcv is not None and t.pair and t.audit]
    selected = st.selectbox("Token to backtest", usable, format_func=lambda a: (
        f"{data.tokens[a].pair.base_symbol if _same(a, data.tokens[a].pair.base_addr) else data.tokens[a].pair.quote_symbol}"
        f"  ({a[:8]}…)")) if usable else None

    out: Optional[BtOutput] = None
    if selected:
        try:
            out = run_backtest(data.tokens[selected], data.cfg.chain, ui["bt"], data.cfg.tf)
        except DataError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest failed: {type(exc).__name__}: {exc}")

    with tab_sec:
        render_security_tab(data, selected)

    with tab_bt:
        if out is not None:
            m = out.metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Net ROI", _fmt_pct(m.get("total_return")))
            k2.metric("Win rate", "—" if "win_rate" not in m else f"{m['win_rate'] * 100:.1f}%")
            k3.metric("Max drawdown", _fmt_pct(m.get("max_drawdown")))
            k4.metric("Total trades executed", int(m.get("n_fills", 0)),
                      help="Executed fills (buys + sells). Round trips: " + str(len(out.trades)))
            if out.blocked:
                st.error("🚫 Security gate CLOSED — this token failed the audit, so all simulated buys were blocked.")
            elif not data.tokens[selected].audit.safe:
                st.warning("⚠️ Gate overridden: this token FAILED the audit but buys are allowed (research mode).")
            else:
                st.success("✅ Security audit passed — simulated buys allowed.")
            for n in out.notes:
                st.caption("ℹ️ " + n)
            st.plotly_chart(build_chart(out, "Price (quote asset per token)"), use_container_width=True)
            st.caption("Historical reserves aren't available, so pool depth is rebuilt from today's pool assuming constant k "
                       "(no LP adds/removes) — liquidity pulls are invisible. Use the haircut slider to stress it. "
                       f"Capital converted at 1 quote asset ≈ ${out.quote_usd:,.2f}.")

    # ---- bottom: execution log
    with st.expander("🧾 Execution log, trades & raw API responses"):
        t1, t2, t3 = st.tabs(["Trade history", "Order / fill log", "Raw JSON responses"])
        with t1:
            if out is not None and out.trades:
                st.dataframe(pd.DataFrame([{
                    "entry": t.entry_ts, "exit": t.exit_ts, "PnL (USD)": round(t.pnl * out.quote_usd, 2),
                    "PnL %": round(t.pnl_pct * 100, 1), "exit slippage (bps)": round(t.exit_slippage_bps),
                    "exit reason": t.tag} for t in out.trades]), use_container_width=True, hide_index=True)
            else:
                st.write("No closed trades.")
        with t2:
            if out is not None and out.exec_log:
                st.dataframe(pd.DataFrame(out.exec_log), use_container_width=True, hide_index=True)
            else:
                st.write("No orders were attempted.")
        with t3:
            # (Streamlit forbids expanders inside expanders, so use a selector instead.)
            if data.log:
                labels = [f"{'✅' if e['ok'] else '❌'} [{e['source']}] {e['url'][:80]} ({e['ms']} ms, {e['at']})"
                          for e in data.log]
                i = st.selectbox("API call", range(len(labels)), format_func=lambda k: labels[k])
                st.json(_trim(data.log[i].get("response", data.log[i].get("error"))))
            else:
                st.write("No API calls recorded.")


# =============================================================================
# Entry points
# =============================================================================
def selftest() -> None:
    """Headless check of the whole data -> audit -> backtest path on the offline demo."""
    for chain in ("solana", "base"):
        world = demo_world(chain, 400, "15m")
        cfg = FetchConfig(chain, world.addresses, True, "15m", 400, AuditLimits())
        data = asyncio.run(run_fetch(cfg))
        for a, td in data.tokens.items():
            assert td.audit and td.ohlcv is not None and td.pair, (a, td.error)
            out = run_backtest(td, chain, BtConfig(), "15m")
            print(f"{chain:7s} {a[:10]}… audit={'PASS' if td.audit.safe else 'FAIL'} "
                  f"gate={'open' if out.gate_open else 'CLOSED'} fills={int(out.metrics['n_fills'])} "
                  f"ROI={out.metrics.get('total_return', 0):+.1%} maxDD={out.metrics.get('max_drawdown', 0):.1%}")
        disc = asyncio.run(run_discovery(chain, True, AuditLimits(), 5000.0, 72.0, 10, "15m"))
        print(f"  discovery: {[(s.token, s.action) for s in disc['suggestions']]} "
              f"rejected={[r['token'] for r in disc['rejected']]}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
