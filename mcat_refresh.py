#!/usr/bin/env python3
"""
MCAT Data Refresh — Daily Macro + Price Pipeline
=================================================
Fetches VIX, SPX, BTC, USDJPY, M2 from free APIs.
Computes BTC 200MA, SPX 200MA, MQS tier classification.
Outputs JSON for Cloudflare KV storage.

Run locally:  python mcat_refresh.py --local
Run in CI:    python mcat_refresh.py   (writes to Cloudflare KV)

Environment variables (for CI):
  CF_ACCOUNT_ID     — Cloudflare account ID
  CF_API_TOKEN      — Cloudflare API token
  CF_KV_NAMESPACE   — KV namespace ID
  FRED_API_KEY      — FRED API key (free, https://fred.stlouisfed.org)
"""

import json
import os
import sys
import time
from datetime import datetime, date, timezone, timedelta

# ── Dependencies ──────────────────────────────────────────────
# pip install yfinance requests
import yfinance as yf
import requests
import numpy as np


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# CoinGecko IDs — Dr. Bob's watchlist
CG_IDS = {
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "XRP":  "ripple",
    "QNT":  "quant-network",
    "TAO":  "bittensor",
    "CC":   "canton-network",
    "XLM":  "stellar",
    "HBAR": "hedera-hashgraph",
    "XDC":  "xdce-crowd-sale",
    "ALGO": "algorand",
    "IOTA": "iota",
    "AERO": "aerodrome-finance",
    "FLR":  "flare-networks",
    "ZORA": "zora",
    "CRO":  "crypto-com-chain",
}

CG_BASE = "https://api.coingecko.com/api/v3"
CG_HEADERS = {"x-cg-demo-api-key": "CG-dJVxAwDLgzBr93UYBNLwqTJZ"}
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# KV key names
KV_KEY_LIVE = "mcat_live_data"
KV_KEY_STATES = "mcat_asset_states"


# ═══════════════════════════════════════════════════════════════
#  DATA FETCHERS
# ═══════════════════════════════════════════════════════════════

def fetch_vix():
    """Fetch current VIX from Yahoo Finance."""
    print("  Fetching VIX...")
    try:
        vix = yf.download("^VIX", period="5d", progress=False)
        if vix.empty:
            raise ValueError("VIX data empty")
        # Handle multi-level columns from yfinance
        if hasattr(vix.columns, 'levels') and len(vix.columns.levels) > 1:
            vix.columns = vix.columns.droplevel(1)
        last = vix.iloc[-1]
        val = float(last["Close"])
        date_str = str(vix.index[-1].date())
        print(f"    VIX = {val:.1f} ({date_str})")
        return {"value": round(val, 1), "date": date_str, "source": "yahoo"}
    except Exception as e:
        print(f"    ⚠️ VIX fetch failed: {e}")
        return {"value": None, "date": None, "source": "yahoo", "error": str(e)}


def fetch_spx():
    """Fetch SPX price + compute 200-day MA."""
    print("  Fetching SPX (S&P 500)...")
    try:
        spx = yf.download("^GSPC", period="300d", progress=False)
        if spx.empty:
            raise ValueError("SPX data empty")
        if hasattr(spx.columns, 'levels') and len(spx.columns.levels) > 1:
            spx.columns = spx.columns.droplevel(1)
        closes = spx["Close"].dropna()
        current = float(closes.iloc[-1])
        date_str = str(spx.index[-1].date())

        # 200-day MA
        if len(closes) >= 200:
            ma200 = float(closes.tail(200).mean())
        else:
            ma200 = float(closes.mean())

        above = current > ma200
        pct = round((current - ma200) / ma200 * 100, 1)
        print(f"    SPX = {current:.0f}, 200MA = {ma200:.0f}, {'above' if above else 'below'} ({pct:+.1f}%)")
        return {
            "value": round(current, 0),
            "ma200": round(ma200, 0),
            "above_ma200": above,
            "pct_from_ma200": pct,
            "date": date_str
        }
    except Exception as e:
        print(f"    ⚠️ SPX fetch failed: {e}")
        return {"value": None, "ma200": None, "above_ma200": None, "pct_from_ma200": None, "date": None, "error": str(e)}


def fetch_btc_macro():
    """Fetch BTC price + compute 200-day MA from CoinGecko."""
    print("  Fetching BTC 200-day history from CoinGecko...")
    try:
        url = f"{CG_BASE}/coins/bitcoin/market_chart?vs_currency=usd&days=250"
        r = requests.get(url, headers=CG_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        prices = [p[1] for p in data["prices"]]
        current = prices[-1]

        if len(prices) >= 200:
            ma200 = sum(prices[-200:]) / 200
        else:
            ma200 = sum(prices) / len(prices)

        above = current > ma200
        pct = round((current - ma200) / ma200 * 100, 1)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(f"    BTC = ${current:,.0f}, 200MA = ${ma200:,.0f}, {'above' if above else 'below'} ({pct:+.1f}%)")
        return {
            "value": round(current, 0),
            "ma200": round(ma200, 0),
            "above_ma200": above,
            "pct_from_ma200": pct,
            "date": date_str
        }
    except Exception as e:
        print(f"    ⚠️ BTC macro fetch failed: {e}")
        return {"value": None, "ma200": None, "above_ma200": None, "pct_from_ma200": None, "date": None, "error": str(e)}


def fetch_usdjpy():
    """Fetch USDJPY + compute 20-day change."""
    print("  Fetching USDJPY...")
    try:
        fx = yf.download("USDJPY=X", period="30d", progress=False)
        if fx.empty:
            raise ValueError("USDJPY data empty")
        if hasattr(fx.columns, 'levels') and len(fx.columns.levels) > 1:
            fx.columns = fx.columns.droplevel(1)
        closes = fx["Close"].dropna()
        current = float(closes.iloc[-1])
        date_str = str(fx.index[-1].date())

        # 20-day change
        if len(closes) >= 20:
            prev20 = float(closes.iloc[-20])
            change_20d = round((current - prev20) / prev20 * 100, 1)
        else:
            change_20d = 0.0

        # Yen unwind detection: USDJPY dropped >= 3% in 20 days
        yen_unwind = change_20d <= -3.0

        print(f"    USDJPY = {current:.1f}, 20d change = {change_20d:+.1f}%, unwind = {yen_unwind}")
        return {
            "value": round(current, 1),
            "change_20d_pct": change_20d,
            "yen_unwind": yen_unwind,
            "date": date_str
        }
    except Exception as e:
        print(f"    ⚠️ USDJPY fetch failed: {e}")
        return {"value": None, "change_20d_pct": None, "yen_unwind": False, "date": None, "error": str(e)}


def fetch_m2():
    """Fetch M2 Money Supply from FRED API."""
    print("  Fetching M2 from FRED...")
    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        print("    ⚠️ FRED_API_KEY not set — skipping M2")
        return {
            "value": None, "yoy_pct": None, "date": None,
            "expanding": False, "contracting": False,
            "error": "FRED_API_KEY not configured"
        }
    try:
        # Fetch last 14 months to compute YoY
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=450)).strftime("%Y-%m-%d")
        params = {
            "series_id": "M2SL",
            "api_key": fred_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "desc",
            "limit": 14
        }
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])

        if len(obs) < 2:
            raise ValueError("Not enough M2 data points")

        # Most recent
        latest = obs[0]
        latest_val = float(latest["value"])
        latest_date = latest["date"]

        # Find value ~12 months prior
        yoy_val = None
        for o in obs:
            d = datetime.strptime(o["date"], "%Y-%m-%d")
            diff = (datetime.strptime(latest_date, "%Y-%m-%d") - d).days
            if 330 <= diff <= 400:
                yoy_val = float(o["value"])
                break

        if yoy_val:
            yoy_pct = round((latest_val - yoy_val) / yoy_val * 100, 1)
        else:
            yoy_pct = None

        expanding = yoy_pct > 2.0 if yoy_pct is not None else False
        contracting = yoy_pct < 0.0 if yoy_pct is not None else False

        print(f"    M2 = ${latest_val:,.0f}B, YoY = {yoy_pct}%, expanding={expanding}")
        return {
            "value": round(latest_val, 0),
            "yoy_pct": yoy_pct,
            "date": latest_date,
            "expanding": expanding,
            "contracting": contracting
        }
    except Exception as e:
        print(f"    ⚠️ M2 fetch failed: {e}")
        return {"value": None, "yoy_pct": None, "date": None, "expanding": False, "contracting": False, "error": str(e)}


def fetch_crypto_prices():
    """Fetch all crypto prices in a single CoinGecko call."""
    print("  Fetching crypto prices from CoinGecko...")
    try:
        ids = ",".join(CG_IDS.values())
        url = f"{CG_BASE}/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true"
        r = requests.get(url, headers=CG_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        prices = {}
        for ticker, cg_id in CG_IDS.items():
            if cg_id in data and "usd" in data[cg_id]:
                prices[ticker] = {
                    "price": data[cg_id]["usd"],
                    "change_24h": round(data[cg_id].get("usd_24h_change", 0) or 0, 1)
                }
                print(f"    {ticker}: ${data[cg_id]['usd']:,.4f} ({prices[ticker]['change_24h']:+.1f}%)")
            else:
                print(f"    ⚠️ {ticker}: not found in response")
                prices[ticker] = {"price": None, "change_24h": None}

        return prices
    except Exception as e:
        print(f"    ⚠️ Crypto price fetch failed: {e}")
        return {t: {"price": None, "change_24h": None} for t in CG_IDS}


# ═══════════════════════════════════════════════════════════════
#  1D DPO (DETRENDED PRICE OSCILLATOR) — Period 7
# ═══════════════════════════════════════════════════════════════

def calc_dpo_normalized(prices_list, period=7, min_periods=60):
    """
    Calculate the normalized DPO oscillator value (0–100 scale).

    DPO = Close - SMA(period, shifted back period/2+1 bars)
    Then normalized to 0-100 using rolling min/max with min_periods.

    Args:
        prices_list: list of daily closing prices (oldest first)
        period: DPO lookback period (default 7 for 1D swing)
        min_periods: minimum history for rolling normalization (default 60)

    Returns:
        float: current DPO value on 0-100 scale, or None if insufficient data
    """
    if len(prices_list) < period + min_periods:
        return None

    prices = np.array(prices_list, dtype=float)
    n = len(prices)

    # Compute SMA
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(prices[i - period + 1:i + 1])

    # DPO = Close - SMA shifted back (period // 2 + 1) bars
    shift = period // 2 + 1
    dpo = np.full(n, np.nan)
    for i in range(shift + period - 1, n):
        dpo[i] = prices[i] - sma[i - shift]

    # Rolling normalization to 0-100 scale
    # Use a window equal to the full available history with min_periods
    valid_dpo = []
    for i in range(n):
        if not np.isnan(dpo[i]):
            valid_dpo.append(dpo[i])

    if len(valid_dpo) < min_periods:
        return None

    # Use all available DPO values for min/max (rolling with expanding window)
    dpo_min = min(valid_dpo)
    dpo_max = max(valid_dpo)

    if dpo_max == dpo_min:
        return 50.0  # flat market

    current_dpo = valid_dpo[-1]
    normalized = (current_dpo - dpo_min) / (dpo_max - dpo_min) * 100
    return round(max(0, min(100, normalized)), 1)


def fetch_1d_dpo(cg_id, ticker_label="", retries=1):
    """
    Fetch 250 days of price history from CoinGecko and compute 1D DPO (period 7).
    Retries once on 429 with a 12-second backoff.
    """
    label = ticker_label or cg_id
    for attempt in range(retries + 1):
        try:
            url = f"{CG_BASE}/coins/{cg_id}/market_chart?vs_currency=usd&days=250"
            r = requests.get(url, headers=CG_HEADERS, timeout=30)
            if r.status_code == 429 and attempt < retries:
                wait = 12
                print(f"    ⏳ {label}: rate limited, waiting {wait}s (retry {attempt+1})...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            prices = [p[1] for p in data["prices"]]

            if len(prices) < 70:
                print(f"    ⚠️ {label}: only {len(prices)} price points, need 70+")
                return {"value": None, "dpo_1w": None, "dpo_2w": None, "in_buy_zone": False, "error": f"insufficient data ({len(prices)} points)"}

            dpo_val = calc_dpo_normalized(prices, period=7, min_periods=60)
            dpo_1w = calc_dpo_normalized(prices, period=20, min_periods=60)
            dpo_2w = calc_dpo_normalized(prices, period=40, min_periods=60)

            if dpo_val is None:
                print(f"    ⚠️ {label}: DPO computation returned None")
                return {"value": None, "dpo_1w": None, "dpo_2w": None, "in_buy_zone": False, "error": "DPO computation failed"}

            in_buy = dpo_val < 20
            print(f"    {label}: 1D DPO = {dpo_val:.1f}, 1W = {dpo_1w}, 2W = {dpo_2w} {'← BUY ZONE' if in_buy else ''}")
            return {"value": dpo_val, "dpo_1w": dpo_1w, "dpo_2w": dpo_2w, "in_buy_zone": in_buy}

        except Exception as e:
            if attempt < retries:
                print(f"    ⏳ {label}: error, retrying in 12s...")
                time.sleep(12)
                continue
            print(f"    ⚠️ {label}: 1D DPO fetch failed: {e}")
            return {"value": None, "dpo_1w": None, "dpo_2w": None, "in_buy_zone": False, "error": str(e)}


# Swing signal exit rules by asset
SWING_EXIT_RULES = {
    "QNT": "fixed-14", "ETH": "fixed-14", "ETC": "fixed-14", "LINK": "fixed-14",
    "XRP": "trend-80", "HBAR": "trend-80", "SOL": "trend-80", "TAO": "trend-80",
    "BNB": "fixed-21"
}


def classify_swing_signal(ticker, dpo_1d, btc_dpo_1d, mqs_tier):
    """
    5-step swing signal decision tree.

    Returns dict with state, confidence, exit_rule, sizing.
    States: INACTIVE, BLOCKED, ACTIVE_HIGH, ACTIVE_STD
    """
    in_buy = dpo_1d is not None and dpo_1d < 20
    btc_cobottom = btc_dpo_1d is not None and btc_dpo_1d < 20
    is_green = mqs_tier == "GREEN"

    if not in_buy:
        return {
            "state": "INACTIVE",
            "dpo_1d": dpo_1d,
            "btc_dpo_1d": btc_dpo_1d,
            "confidence": None,
            "exit_rule": None,
            "sizing": None
        }

    if is_green:
        return {
            "state": "BLOCKED",
            "dpo_1d": dpo_1d,
            "btc_dpo_1d": btc_dpo_1d,
            "confidence": None,
            "exit_rule": None,
            "sizing": None,
            "reason": "MQS GREEN — swing signals skipped in calm markets"
        }

    exit_rule = SWING_EXIT_RULES.get(ticker, "fixed-14")

    if btc_cobottom:
        return {
            "state": "ACTIVE_HIGH",
            "dpo_1d": dpo_1d,
            "btc_dpo_1d": btc_dpo_1d,
            "confidence": "HIGH",
            "exit_rule": exit_rule,
            "sizing": "full"
        }
    else:
        return {
            "state": "ACTIVE_STD",
            "dpo_1d": dpo_1d,
            "btc_dpo_1d": btc_dpo_1d,
            "confidence": "STANDARD",
            "exit_rule": exit_rule,
            "sizing": "half"
        }


# ═══════════════════════════════════════════════════════════════
#  AUTOMATION ENGINE — Signal Detection, Cycle Phases, Exits
# ═══════════════════════════════════════════════════════════════

DEFAULT_STATE = {
    'signal_active': False,
    'signal_date': None,
    'signal_days_ago': 0,
    'cluster_days': 0,
    'dpo20_at_day5': None,
    'right_translation': None,
    'rt_check_date': None,
    'cycle_phase': 'WAITING',
    'prev_phase': 'WAITING',
    'was_overheated': False,
    'exit_level': None,
    'peak_dpo40': None,
    'peak_dpo40_date': None,
}

BOOTSTRAP = {
    'BTC':  {'signal_active': True, 'signal_date': '2026-02-13', 'cycle_phase': 'EARLY_SIGNAL'},
    'ETH':  {'signal_active': True, 'signal_date': '2026-02-06', 'cycle_phase': 'EARLY_SIGNAL'},
    'XRP':  {'signal_active': True, 'signal_date': '2026-02-06', 'cycle_phase': 'EARLY_SIGNAL'},
    'BNB':  {'signal_active': True, 'signal_date': '2026-02-06', 'cycle_phase': 'EARLY_SIGNAL'},
    'DASH': {'signal_active': True, 'signal_date': '2026-02-13', 'cycle_phase': 'EARLY_SIGNAL'},
    'XLM':  {'signal_active': True, 'signal_date': '2026-01-01', 'cycle_phase': 'ACCUMULATION'},
}


def detect_signal(asset_state, dpo20_today, dpo40_today):
    """
    Auto-detect signal date. When both DPO-20 and DPO-40 < 20 for 2+
    consecutive days, fire a new signal. 30-day cooldown between signals.
    """
    THRESHOLD = 20
    MIN_CLUSTER = 2
    COOLDOWN_DAYS = 30

    both_below = (dpo20_today is not None and dpo40_today is not None
                  and dpo20_today < THRESHOLD and dpo40_today < THRESHOLD)
    today = date.today()

    if both_below:
        asset_state['cluster_days'] = asset_state.get('cluster_days', 0) + 1

        if asset_state['cluster_days'] >= MIN_CLUSTER:
            # Check 30-day cooldown
            last_signal = asset_state.get('signal_date')
            if last_signal:
                days_since = (today - date.fromisoformat(last_signal)).days
                if days_since < COOLDOWN_DAYS:
                    return asset_state

            if not asset_state.get('signal_active'):
                # NEW SIGNAL FIRED
                signal_start = today - timedelta(days=asset_state['cluster_days'] - 1)
                asset_state['signal_active'] = True
                asset_state['signal_date'] = signal_start.isoformat()
                asset_state['dpo20_at_day5'] = None
                asset_state['right_translation'] = None
                asset_state['rt_check_date'] = None
                asset_state['exit_level'] = None
                asset_state['peak_dpo40'] = dpo40_today
                asset_state['was_overheated'] = False
    else:
        asset_state['cluster_days'] = 0

    if asset_state.get('signal_active') and asset_state.get('signal_date'):
        asset_state['signal_days_ago'] = (today - date.fromisoformat(asset_state['signal_date'])).days

    return asset_state


def check_right_translation(asset_state, dpo20_today):
    """
    Store DPO-20 at day 5, evaluate at day 20. Computed once per signal.
    +45.6 percentage points EV difference between RT YES vs NO.
    """
    if not asset_state.get('signal_active') or not asset_state.get('signal_date'):
        return asset_state

    days_since = asset_state.get('signal_days_ago', 0)

    # Store DPO-20 at day 5 (±1 day tolerance)
    if days_since in [4, 5, 6] and asset_state.get('dpo20_at_day5') is None:
        asset_state['dpo20_at_day5'] = dpo20_today

    # Evaluate at day 20 (±1 day tolerance)
    if days_since in [19, 20, 21] and asset_state.get('right_translation') is None:
        day5_value = asset_state.get('dpo20_at_day5')
        if day5_value is not None:
            asset_state['right_translation'] = (dpo20_today > day5_value)
            asset_state['rt_check_date'] = date.today().isoformat()

    return asset_state


def update_cycle_phase(asset_state, dpo20_today, dpo40_today):
    """
    7-state machine: WAITING, EARLY_SIGNAL, ACCUMULATION, RISING,
    OVERHEATED, COOLING, TOPPED. Call daily after detect_signal() and
    check_right_translation().
    """
    prev = asset_state.get('cycle_phase', 'WAITING')
    was_oh = asset_state.get('was_overheated', False)
    signal_active = asset_state.get('signal_active', False)

    # Guard against None values
    d20 = dpo20_today if dpo20_today is not None else 50.0
    d40 = dpo40_today if dpo40_today is not None else 50.0

    if d40 > 80 or d20 > 80:
        new_phase = 'OVERHEATED'
        was_oh = True

    elif was_oh and d40 < 60 and d20 < 60:
        new_phase = 'TOPPED'
        was_oh = False

    elif was_oh and (d40 >= 60 or d20 >= 60):
        new_phase = 'COOLING'

    elif d20 < 20 and d40 < 20:
        new_phase = 'ACCUMULATION'

    elif d20 < 25 or d40 < 25:
        new_phase = 'EARLY_SIGNAL'

    elif signal_active and prev in ('ACCUMULATION', 'EARLY_SIGNAL', 'RISING') and d20 > 30:
        new_phase = 'RISING'

    else:
        new_phase = 'WAITING'

    # TOPPED → WAITING = cycle officially over
    if prev == 'TOPPED' and new_phase == 'WAITING':
        asset_state['signal_active'] = False

    asset_state['prev_phase'] = prev
    asset_state['cycle_phase'] = new_phase
    asset_state['was_overheated'] = was_oh

    return asset_state


def update_exit_level(asset_state, dpo20_today, dpo40_today, symbol):
    """
    3-level exit framework + HBAR auto-exit. Call daily after
    update_cycle_phase().
    """
    if not asset_state.get('signal_active'):
        asset_state['exit_level'] = None
        return asset_state

    phase = asset_state.get('cycle_phase')
    d20 = dpo20_today if dpo20_today is not None else 50.0
    d40 = dpo40_today if dpo40_today is not None else 50.0

    # Track peak DPO-40 this cycle
    peak = asset_state.get('peak_dpo40') or 0
    if d40 > peak:
        asset_state['peak_dpo40'] = d40
        asset_state['peak_dpo40_date'] = date.today().isoformat()

    # HBAR SPECIAL CASE: auto-exit at DPO>80
    if symbol == 'HBAR' and (d40 > 80 or d20 > 80):
        asset_state['exit_level'] = 'AUTO_EXIT_HBAR'
        return asset_state

    # Level 1: OVERHEATED (DPO > 80)
    if phase == 'OVERHEATED':
        asset_state['exit_level'] = 'OVERHEATED'
        return asset_state

    # Level 2 + 3: Only if peak exceeded 80 this cycle
    if asset_state.get('peak_dpo40', 0) > 80:
        if d40 < 50 and d20 < 50:
            # Level 3: CYCLE_COMPLETE
            asset_state['exit_level'] = 'CYCLE_COMPLETE'
            asset_state['signal_active'] = False
        elif d40 < 60 and d20 < 60:
            # Level 2: TAKE_PROFITS
            asset_state['exit_level'] = 'TAKE_PROFITS'
        else:
            asset_state['exit_level'] = 'OVERHEATED'
        return asset_state

    asset_state['exit_level'] = None
    return asset_state


def process_asset_automation(symbol, dpo7, dpo20, dpo40, prev_state=None):
    """
    Run full automation pipeline for one asset. ORDER MATTERS.
    prev_state comes from KV read; None triggers bootstrap/default.
    """
    if prev_state is None:
        state = {**DEFAULT_STATE, **BOOTSTRAP.get(symbol, {})}
    else:
        state = prev_state

    # Run automation pipeline in order
    state = detect_signal(state, dpo20, dpo40)
    state = check_right_translation(state, dpo20)
    state = update_cycle_phase(state, dpo20, dpo40)
    state = update_exit_level(state, dpo20, dpo40, symbol)

    # Merge DPO values into state
    state['dpo_7'] = round(dpo7, 1) if dpo7 is not None else None
    state['dpo_1w'] = round(dpo20, 1) if dpo20 is not None else None
    state['dpo_2w'] = round(dpo40, 1) if dpo40 is not None else None

    return state


# ═══════════════════════════════════════════════════════════════
#  MQS TIER CLASSIFICATION (macro_score_v3 / M65)
# ═══════════════════════════════════════════════════════════════

def classify_mqs(vix_val, btc_above_200ma, spx_co_bottom, yen_unwind, m2_expanding, m2_contracting):
    """
    MQS Lookup Table (M65) — first match wins.
    Returns dict with tier, rule, description, sizing info.
    """
    vix = vix_val if vix_val is not None else 21.0  # fallback
    bull = btc_above_200ma
    bear = not bull
    co_bot = spx_co_bottom
    yen_uw = yen_unwind
    m2_exp = m2_expanding
    m2_con = m2_contracting

    # Rule 1: VIX >45 → GREEN capitulation
    if vix > 45:
        tier, rule, name = "GREEN", 1, "VIX >45 capitulation"
        ev, hit, n, sizing = 111.7, 100, 4, "80–100%"

    # Rule 2: SPX co-bottom + VIX <30 → RED
    elif co_bot and vix < 30:
        tier, rule, name = "RED", 2, "SPX co-bottom + VIX <30"
        ev, hit, n, sizing = 7.8, 55, 11, "20–25%"

    # Rule 3: VIX 20-30 + BTC bear → RED
    elif vix >= 20 and vix <= 30 and bear:
        tier, rule, name = "RED", 3, "VIX 20–30 + BTC bear"
        ev, hit, n, sizing = 15.3, 65, 34, "20–25%"

    # Rule 4: VIX 20-30 + BTC bull → AMBER
    elif vix >= 20 and vix <= 30 and bull:
        tier, rule, name = "AMBER", 4, "VIX 20–30 + BTC bull"
        ev, hit, n, sizing = 28.4, 79, 29, "50–60%"

    # Rule 5: VIX 30-45 → AMBER
    elif vix > 30 and vix <= 45:
        tier, rule, name = "AMBER", 5, "VIX 30–45"
        ev, hit, n, sizing = 27.6, 73, 15, "50–60%"

    # Rule 6: VIX <20 + BTC bear + no co-bottom → AMBER
    elif vix < 20 and bear and not co_bot:
        tier, rule, name = "AMBER", 6, "VIX <20 + BTC bear"
        ev, hit, n, sizing = 26.5, 75, 20, "50–60%"

    # Rule 7: VIX <20 + BTC bull + no co-bottom → GREEN (Calm)
    elif vix < 20 and bull and not co_bot:
        tier, rule, name = "GREEN", 7, "VIX <20 + BTC bull (Calm)"
        ev, hit, n, sizing = 155.5, 58, 40, "70–80%"

        # Override 8: Yen unwind → downgrade to AMBER
        if yen_uw:
            tier, rule, name = "AMBER", 8, "GREEN + yen unwind override"
            ev, hit, n, sizing = 19.5, 65, 8, "50–60%"
        # Override 9: M2 expanding → premium GREEN
        elif m2_exp:
            tier, rule, name = "GREEN", 9, "GREEN + M2 expanding (premium)"
            ev, hit, n, sizing = 207.9, 62, 16, "70–80%"
        # Override 10: M2 contracting → size as AMBER
        elif m2_con:
            tier, rule, name = "GREEN", 10, "GREEN + M2 contracting (reduced)"
            ev, hit, n, sizing = 10.9, 50, 10, "50–60%"

    else:
        # Fallback — should not normally reach here
        tier, rule, name = "AMBER", 0, "Unclassified (fallback)"
        ev, hit, n, sizing = 20.0, 65, 0, "40–50%"

    return {
        "tier": tier,
        "rule": f"Rule {rule}",
        "rule_number": rule,
        "description": name,
        "ev_90d": ev,
        "hit_rate": hit,
        "sample_n": n,
        "sizing": sizing
    }


def compute_confidence(dpo_both_below_20, btc_above_200ma, spx_above_200ma, right_translation="pending"):
    """Confidence score: 4 possible points."""
    score = 0
    components = {}

    components["dpo_both_below_20"] = dpo_both_below_20
    if dpo_both_below_20:
        score += 1

    components["btc_above_200ma"] = btc_above_200ma
    if btc_above_200ma:
        score += 1

    components["spx_above_200ma"] = spx_above_200ma
    if spx_above_200ma:
        score += 1

    components["right_translation"] = right_translation
    if right_translation == True:
        score += 1

    return {"score": score, "max": 4, "components": components}


def detect_vix_boundary(vix_val):
    """Detect if VIX is near a critical threshold."""
    if vix_val is None:
        return {"near_boundary": False, "note": "VIX data unavailable"}

    notes = []
    near = False

    if 18.0 <= vix_val <= 22.0:
        near = True
        if vix_val < 20:
            notes.append(
                f"VIX at {vix_val:.1f} — near the 20 threshold. "
                f"Tier may flip to RED if VIX closes above 20."
            )
        else:
            notes.append(
                f"VIX at {vix_val:.1f} — near the 20 threshold. "
                f"Tier may flip to AMBER if VIX closes below 20."
            )

    if 28.0 <= vix_val <= 32.0:
        near = True
        notes.append(f"VIX at {vix_val:.1f} — near the 30 threshold (AMBER/extreme boundary).")

    if 43.0 <= vix_val <= 47.0:
        near = True
        notes.append(f"VIX at {vix_val:.1f} — near the 45 threshold (capitulation GREEN trigger).")

    return {
        "near_boundary": near,
        "note": " ".join(notes) if notes else "VIX stable, not near any threshold."
    }


# ═══════════════════════════════════════════════════════════════
#  CLOUDFLARE KV WRITER
# ═══════════════════════════════════════════════════════════════

def write_to_kv(data):
    """Write JSON to Cloudflare KV via REST API."""
    return write_kv_key(KV_KEY_LIVE, data)


def read_kv_key(key):
    """Read a JSON value from Cloudflare KV. Returns None on failure."""
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    namespace_id = os.environ.get("CF_KV_NAMESPACE")

    if not all([account_id, api_token, namespace_id]):
        return None

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{key}"
    )
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 404:
            return None  # Key doesn't exist yet
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ⚠️ KV read ({key}) failed: {e}")
        return None


def write_kv_key(key, data):
    """Write JSON to a specific Cloudflare KV key."""
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    namespace_id = os.environ.get("CF_KV_NAMESPACE")

    if not all([account_id, api_token, namespace_id]):
        print(f"\n⚠️  Cloudflare credentials not configured — skipping KV write ({key}).")
        print("    Set CF_ACCOUNT_ID, CF_API_TOKEN, CF_KV_NAMESPACE as environment variables.")
        return False

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{key}"
    )
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.put(url, headers=headers, data=json.dumps(data, cls=NumpyEncoder), timeout=30)
        r.raise_for_status()
        result = r.json()
        if result.get("success"):
            print(f"\n✅ Written to KV ({key}) successfully.")
            return True
        else:
            print(f"\n❌ KV write ({key}) failed: {result.get('errors', 'unknown')}")
            return False
    except Exception as e:
        print(f"\n❌ KV write ({key}) error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(local_mode=False):
    """Run the full data refresh pipeline."""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  MCAT Data Refresh Pipeline")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Mode: {'LOCAL (file output)' if local_mode else 'CI (Cloudflare KV)'}")
    print(f"{'='*60}\n")

    errors = []

    # ── Fetch all data ────────────────────────────────────────
    print("[1/5] Macro indicators...")
    vix = fetch_vix()
    time.sleep(1)  # rate limit courtesy

    spx = fetch_spx()
    time.sleep(1)

    usdjpy = fetch_usdjpy()
    time.sleep(2)  # CoinGecko rate limit

    print("\n[2/5] BTC 200-day data...")
    btc = fetch_btc_macro()
    time.sleep(2)

    print("\n[3/5] M2 Money Supply...")
    m2 = fetch_m2()
    time.sleep(1)

    print("\n[4/5] Crypto prices...")
    prices = fetch_crypto_prices()

    # ── Fetch 1D DPO for BTC + all assets ─────────────────────
    print("\n[4b/5] Computing 1D DPO (swing signal)...")
    time.sleep(2)  # CoinGecko rate limit

    # BTC 1D DPO — reuse the 250-day data already fetched for BTC macro
    try:
        url = f"{CG_BASE}/coins/bitcoin/market_chart?vs_currency=usd&days=250"
        r = requests.get(url, headers=CG_HEADERS, timeout=30)
        r.raise_for_status()
        btc_prices = [p[1] for p in r.json()["prices"]]
        btc_dpo_val = calc_dpo_normalized(btc_prices, period=7, min_periods=60)
        btc_dpo_1w = calc_dpo_normalized(btc_prices, period=20, min_periods=60)
        btc_dpo_2w = calc_dpo_normalized(btc_prices, period=40, min_periods=60)
        btc_dpo = {"value": btc_dpo_val, "dpo_1w": btc_dpo_1w, "dpo_2w": btc_dpo_2w, "in_buy_zone": btc_dpo_val is not None and btc_dpo_val < 20}
        print(f"    BTC: 1D DPO = {btc_dpo_val:.1f}, 1W = {btc_dpo_1w}, 2W = {btc_dpo_2w} {'← BUY ZONE' if btc_dpo['in_buy_zone'] else ''}")
    except Exception as e:
        print(f"    ⚠️ BTC 1D DPO failed: {e}")
        btc_dpo = {"value": None, "dpo_1w": None, "dpo_2w": None, "in_buy_zone": False, "error": str(e)}
    btc_dpo_val = btc_dpo.get("value")

    # All other assets — rate-limited with retry
    asset_dpos = {"BTC": btc_dpo}
    for ticker, cg_id in CG_IDS.items():
        if ticker == "BTC":
            continue
        time.sleep(6)  # CoinGecko free tier: generous spacing with API key
        asset_dpos[ticker] = fetch_1d_dpo(cg_id, ticker)

    # ── Track fetch errors ────────────────────────────────────
    for name, obj in [("VIX", vix), ("SPX", spx), ("BTC", btc), ("USDJPY", usdjpy), ("M2", m2)]:
        if obj.get("error"):
            errors.append(f"{name}: {obj['error']}")

    # ── Classify MQS tier ─────────────────────────────────────
    print("\n[5/5] Computing MQS tier...")

    btc_above = btc.get("above_ma200", False)
    spx_above = spx.get("above_ma200", False)
    # SPX co-bottom: SPX below 200MA (proxy — in production, check ±30d of signal)
    spx_co_bottom = not spx_above if spx.get("above_ma200") is not None else False
    yen_unwind = usdjpy.get("yen_unwind", False)
    m2_expanding = m2.get("expanding", False)
    m2_contracting = m2.get("contracting", False)

    mqs = classify_mqs(
        vix_val=vix.get("value"),
        btc_above_200ma=btc_above,
        spx_co_bottom=spx_co_bottom,
        yen_unwind=yen_unwind,
        m2_expanding=m2_expanding,
        m2_contracting=m2_contracting
    )

    # VIX boundary detection
    vix_boundary = detect_vix_boundary(vix.get("value"))
    mqs["vix_near_boundary"] = vix_boundary["near_boundary"]
    mqs["vix_boundary_note"] = vix_boundary["note"]

    # Confidence score
    # DPO both below 20 — this comes from the dashboard, not from this pipeline.
    # For now, we set it as True based on the current state (signal date Feb 13, cycle active).
    # The dashboard will override this with its own live DPO calculation.
    confidence = compute_confidence(
        dpo_both_below_20=True,  # Dashboard overrides this
        btc_above_200ma=btc_above,
        spx_above_200ma=spx_above,
        right_translation="pending"  # Manual assessment
    )

    print(f"\n  MQS Result: {mqs['tier']} ({mqs['description']})")
    print(f"  Confidence: {confidence['score']}/{confidence['max']}")
    print(f"  VIX Boundary: {'⚠️ YES' if vix_boundary['near_boundary'] else 'No'}")
    if errors:
        print(f"\n  ⚠️ Fetch errors ({len(errors)}):")
        for e in errors:
            print(f"    • {e}")

    # ── Assemble output JSON ──────────────────────────────────
    # Classify swing signals for each asset
    swing_signals = {}
    for ticker in CG_IDS:
        dpo_val = asset_dpos.get(ticker, {}).get("value")
        swing_signals[ticker] = classify_swing_signal(ticker, dpo_val, btc_dpo_val, mqs["tier"])

    active_swings = [t for t, s in swing_signals.items() if s["state"].startswith("ACTIVE")]
    if active_swings:
        print(f"\n  🔔 Active swing signals: {', '.join(active_swings)}")
    else:
        print(f"\n  No active swing signals.")

    # ── Automation: Signal Detection, Cycle Phases, Exits ─────
    print("\n[6/6] Running automation engine...")

    # Read previous asset states from KV (single key)
    prev_states = None
    if not local_mode:
        prev_states = read_kv_key(KV_KEY_STATES)
    elif os.path.exists("mcat_asset_states.json"):
        try:
            with open("mcat_asset_states.json") as f:
                prev_states = json.load(f)
        except Exception:
            prev_states = None

    if prev_states is None:
        print("    No previous states found — using bootstrap/defaults (first run)")

    asset_states = {}
    for ticker in CG_IDS:
        dpo_data = asset_dpos.get(ticker, {})
        dpo7 = dpo_data.get("value")
        dpo20 = dpo_data.get("dpo_1w")
        dpo40 = dpo_data.get("dpo_2w")

        # Get previous state for this asset (None on first run triggers bootstrap)
        prev = prev_states.get(ticker) if prev_states else None

        state = process_asset_automation(ticker, dpo7, dpo20, dpo40, prev)
        asset_states[ticker] = state

        # Log significant states
        phase = state.get('cycle_phase', 'WAITING')
        if phase != 'WAITING':
            sig_date = state.get('signal_date', '?')
            days = state.get('signal_days_ago', 0)
            rt = state.get('right_translation')
            exit_lv = state.get('exit_level')
            rt_str = f"RT={'YES' if rt else 'NO'}" if rt is not None else "RT=pending"
            exit_str = f" EXIT={exit_lv}" if exit_lv else ""
            print(f"    {ticker}: {phase} (signal {sig_date}, day {days}, {rt_str}{exit_str})")

    # Write asset states to KV
    if not local_mode:
        write_kv_key(KV_KEY_STATES, asset_states)
    with open("mcat_asset_states.json", "w") as f:
        json.dump(asset_states, f, indent=2, cls=NumpyEncoder)

    output = {
        "last_updated": now.isoformat(),
        "stale": len(errors) > 2,  # Mark stale if >2 sources failed
        "fetch_errors": errors,
        "macro": {
            "vix": vix,
            "spx": spx,
            "btc": btc,
            "usdjpy": usdjpy,
            "m2": m2,
            "btc_1d_dpo": btc_dpo
        },
        "mqs": mqs,
        "confidence": confidence,
        "prices": {t: p["price"] for t, p in prices.items()},
        "price_changes_24h": {t: p["change_24h"] for t, p in prices.items()},
        "asset_dpo_1d": {t: d for t, d in asset_dpos.items()},
        "swing_signals": swing_signals,
        "asset_states": asset_states
    }

    # ── Output ────────────────────────────────────────────────
    if local_mode:
        out_path = "mcat_live_data.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f"\n✅ Local output: {out_path} ({os.path.getsize(out_path):,} bytes)")
    else:
        write_to_kv(output)

    # Also save locally as backup regardless
    with open("mcat_live_data.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Tier: {mqs['tier']} | VIX: {vix.get('value', '?')} | BTC: ${btc.get('value', '?'):,.0f}")
    print(f"{'='*60}\n")

    return output


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    local = "--local" in sys.argv
    run_pipeline(local_mode=local)
