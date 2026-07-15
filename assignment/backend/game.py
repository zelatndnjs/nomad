"""트레이딩 연습 게임 — 과거 특정 시점까지만 보여주고, 사용자가 진입/손절/익절을 정하면
미래를 재생해 결과(익절/손절/미체결)와 수익률을 계산한다. (롱 전용 v1)"""
import random

import pandas as pd

from .data import fetch_klines
from .strategies import compute_timing


def _candles(raw):
    return [{"time": int(k[0]) // 1000, "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4])} for k in raw]


def _df(candles):
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("date")[["open", "high", "low", "close"]]


def new_game(symbol="BTCUSDT", interval="4h", strategy_key="sma_cross", past=180, future=150):
    """랜덤 과거 시점을 골라, 그 시점까지의 캔들 + 전략의 진입 근거/제안 손절·익절을 반환."""
    candles = _candles(fetch_klines(symbol, interval, 1500))
    if len(candles) < past + future + 10:
        past = max(60, len(candles) // 3)
        future = max(40, len(candles) // 4)
    t = random.randint(past, len(candles) - future - 1)
    setup = candles[t - past:t + 1]
    entry = candles[t]
    tim = compute_timing(_df(setup), strategy_key)   # 진입 시점까지의 데이터로 판단
    return {
        "game_id": f"{symbol}|{interval}|{entry['time']}",
        "symbol": symbol, "interval": interval,
        "candles": setup,
        "entry_price": round(entry["close"], 2),
        "entry_time": entry["time"],
        "strategy_name": tim["strategy_name"],
        "signal": tim["signal"],
        "entry_reason": tim["entry_reason"],
        "suggest_sl": round(tim["stop_loss"], 2),
        "suggest_tp": round(tim["take_profit"], 2),
        "sl_method": tim["sl_method"],
        "tp_method": tim["tp_method"],
    }


def walk_forward(fut, entry_price, entry_time, tp, sl):
    """미래 캔들을 순서대로 걸어가며 손절/익절 중 먼저 닿는 것을 판정 (순수 함수, 테스트 가능).
    한 봉에서 둘 다 닿으면 손절 우선(보수적). 아무것도 안 닿으면 마지막 종가로 청산(timeout)."""
    if not fut:
        return "timeout", entry_price, entry_time
    outcome, exit_price, exit_time = "timeout", fut[-1]["close"], fut[-1]["time"]
    for c in fut:
        if c["low"] <= sl:
            return "sl", sl, c["time"]
        if c["high"] >= tp:
            return "tp", tp, c["time"]
    return outcome, exit_price, exit_time


def simulate(symbol, interval, entry_time, entry_price, tp, sl, future=150):
    """진입 시점 이후 미래 캔들을 걸어가며 손절/익절 중 무엇이 먼저 닿는지 판정."""
    candles = _candles(fetch_klines(symbol, interval, 1500))
    idx = next((i for i, c in enumerate(candles) if c["time"] == entry_time), None)
    if idx is None:
        idx = min(range(len(candles)), key=lambda i: abs(candles[i]["time"] - entry_time))
    fut = candles[idx + 1: idx + 1 + future]

    outcome, exit_price, exit_time = walk_forward(fut, entry_price, entry_time, tp, sl)
    pnl = (exit_price / entry_price - 1) * 100
    bh = ((fut[-1]["close"] / entry_price - 1) * 100) if fut else 0.0
    return {
        "future": fut,
        "outcome": outcome,
        "exit_price": round(exit_price, 2),
        "exit_time": exit_time,
        "pnl_pct": pnl,
        "bh_pct": bh,
        "rr": (tp - entry_price) / (entry_price - sl) if entry_price > sl else 0.0,
    }
