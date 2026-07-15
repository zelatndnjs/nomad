"""오프라인 단위 테스트 — 합성 데이터로 전략·백테스트·시그널빌더·지표·게임·판정 검증.
네트워크(Binance)·LLM 없이 돈다.  실행: uv run pytest -q"""
import numpy as np
import pandas as pd
import pytest

from backend import strategies as S
from backend.game import walk_forward
from backend.research import is_valid


@pytest.fixture
def df():
    """200봉 합성 OHLC (상승 100 + 하락 100 + 잔물결)."""
    n = 200
    close = np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 120, 100)])
    close = close + np.sin(np.arange(n)) * 2
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": close, "high": close + 3, "low": close - 3, "close": close}, index=idx)


# ---------- 전략 · 백테스트 ----------
def test_all_strategy_signals_are_binary(df):
    for key, strat in S.STRATEGY_LIBRARY.items():
        pos = strat["signal"](df)
        assert len(pos) == len(df)
        assert set(pd.unique(pos)) <= {0, 1}, f"{key} 포지션이 0/1 아님"


def test_run_backtest_shape(df):
    r = S.run_backtest(df, S.sig_sma_cross(df))
    for k in ["total_return", "bh_return", "mdd", "n_trades", "win_rate"]:
        assert k in r
    assert isinstance(r["total_return"], float)
    assert r["mdd"] <= 0
    assert 0.0 <= r["win_rate"] <= 1.0


def test_backtest_always_hold_equals_buyhold():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    close = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190.0], index=idx)
    d = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    r = S.run_backtest(d, pd.Series(1, index=idx))
    assert abs(r["total_return"] - r["bh_return"]) < 1e-9


# ---------- 범용 시그널 빌더 (리서치 전략 실행 엔진) ----------
def test_series_resolver(df):
    assert S._series(df, "close").equals(df.close)
    assert (S._series(df, "const:30") == 30).all()
    assert len(S._series(df, "rsi:14")) == len(df)


def test_cond_and_build_signal(df):
    cfg = {"entry": {"a": "sma:5", "op": "cross_above", "b": "sma:20"},
           "exit": {"a": "sma:5", "op": "cross_below", "b": "sma:20"}}
    pos = S.build_signal(df, cfg)
    assert len(pos) == len(df)
    assert set(pd.unique(pos)) <= {0, 1}


def test_build_signal_bad_ref_raises(df):
    with pytest.raises(ValueError):
        S.build_signal(df, {"entry": {"a": "nope:1", "op": "above", "b": "close"},
                            "exit": {"a": "close", "op": "below", "b": "close"}})


# ---------- 지표 ----------
def test_overlay_and_pane_indicators(df):
    ov = S.overlay_lines(df, ["SMA 10", "볼린저밴드"])
    assert len(ov) == 4                      # SMA10 + 밴드 3선
    assert all("data" in o and "color" in o for o in ov)
    panes = S.pane_indicators(df, ["RSI", "MACD"])
    assert [p["id"] for p in panes] == ["rsi", "macd"]
    assert len(panes[0]["lines"]) == 3       # RSI + 70/30 가이드


def test_custom_strategy_has_entry_reason(df):
    cfg = {"key": "t_custom", "name": "테스트전략", "desc": "x",
           "entry": {"a": "sma:5", "op": "cross_above", "b": "sma:20"},
           "exit": {"a": "sma:5", "op": "cross_below", "b": "sma:20"}, "indicators": ["SMA 10"]}
    S.register_custom(cfg)
    try:
        reason = S.current_condition(df, "t_custom")
        assert reason and "진입 규칙" in reason      # 커스텀도 근거 채워짐 (갭 수정 확인)
        assert S.default_indicators("t_custom") == ["SMA 10"]
    finally:
        S.STRATEGY_LIBRARY.pop("t_custom", None)


# ---------- 게임 시뮬 (walk_forward 순수 함수) ----------
def _bar(t, lo, hi, cl):
    return {"time": t, "low": lo, "high": hi, "close": cl}


def test_walk_forward_outcomes():
    assert walk_forward([_bar(1, 99, 101, 100), _bar(2, 98, 112, 110)], 100, 0, 110, 95)[0] == "tp"
    assert walk_forward([_bar(1, 90, 101, 95)], 100, 0, 110, 95)[0] == "sl"
    assert walk_forward([_bar(1, 99, 101, 100)], 100, 0, 110, 95)[0] == "timeout"
    assert walk_forward([_bar(1, 90, 115, 100)], 100, 0, 110, 95)[0] == "sl"   # 한 봉 둘 다 → 손절 우선
    assert walk_forward([], 100, 7, 110, 95) == ("timeout", 100, 7)            # 미래 없음


# ---------- 리서치 판정 ----------
def test_is_valid_thresholds():
    assert is_valid({"ret": 0.5, "rpm": 2.0, "trades": 10}) is True
    assert is_valid({"ret": 0.5, "rpm": 1.0, "trades": 10}) is False   # 수익/MDD < 1.5
    assert is_valid({"ret": -0.1, "rpm": 2.0, "trades": 10}) is False  # 수익 음수
    assert is_valid({"ret": 0.5, "rpm": 2.0, "trades": 0}) is False    # 매매 없음
