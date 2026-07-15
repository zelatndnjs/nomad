"""기술적 지표 · 5개 전략 · 백테스트 · 타이밍(진입/손절/익절) 계산."""
import pandas as pd
import numpy as np


# ---------- 지표 ----------
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _hold(entry, exit):
    e, x = entry.fillna(False), exit.fillna(False)
    pos = pd.Series(0, index=entry.index)
    h = 0
    for i in range(len(pos)):
        if h == 0 and e.iloc[i]:
            h = 1
        elif h == 1 and x.iloc[i]:
            h = 0
        pos.iloc[i] = h
    return pos


# ---------- 5개 전략 (position 시계열: 1=보유, 0=현금) ----------
def sig_sma_cross(df, fast=10, slow=30):
    # 백테스트 결과 20/50 보다 10/30 이 수익·위험대비 모두 우수해 조정.
    f, s = sma(df.close, fast), sma(df.close, slow)
    return ((f > s) & f.notna() & s.notna()).astype(int)


def sig_rsi(df, n=10, low=40, high=70, ma=100):
    # 단순 RSI 역추세는 추세장에서 부진 → '상승추세(장기 이평 위)에서 RSI 과매도 눌림목 매수'로 조정.
    r = rsi(df.close, n)
    uptrend = df.close > sma(df.close, ma)
    return _hold((r < low) & uptrend, r > high)


def sig_macd(df, fast=12, slow=26, sig=9):
    m = ema(df.close, fast) - ema(df.close, slow)
    sg = ema(m, sig)
    pos = (m > sg).astype(int)
    pos[m.isna() | sg.isna()] = 0
    return pos


def sig_bollinger(df, n=20, k=2):
    ma, sd = sma(df.close, n), df.close.rolling(n).std()
    return _hold(df.close < (ma - k * sd), df.close > ma)


def sig_breakout(df, n=20):
    return _hold(df.close > df.close.rolling(n).max().shift(1),
                 df.close < df.close.rolling(n).min().shift(1))


STRATEGY_LIBRARY = {
    "sma_cross": {"name": "골든/데드 크로스 (SMA 10/30)",
                  "desc": "단기 이동평균(10)이 장기 이동평균(30)을 상향 돌파하면 매수(골든크로스), 하향 돌파하면 매도(데드크로스). 대표적 추세추종. (백테스트로 20/50 → 10/30 조정)",
                  "signal": sig_sma_cross},
    "rsi": {"name": "RSI 눌림목 (상승추세 과매도 매수)",
            "desc": "장기 이동평균(100) 위 상승추세에서 RSI가 과매도(40 아래)로 눌리면 매수, 과매수(70 위)면 익절. (단순 RSI 역추세가 추세장에서 부진해 트렌드 필터로 조정)",
            "signal": sig_rsi},
    "macd": {"name": "MACD 시그널 크로스 (12/26/9)",
             "desc": "MACD선이 시그널선을 상향 돌파하면 매수, 하향 돌파하면 매도. 추세+모멘텀.",
             "signal": sig_macd},
    "bollinger": {"name": "볼린저 밴드 (20, 2σ)",
                  "desc": "가격이 하단밴드 아래로 가면 매수, 중심선(이평) 회복 시 매도. 평균회귀.",
                  "signal": sig_bollinger},
    "breakout": {"name": "돌파 (Donchian 20일)",
                 "desc": "최근 20일 고가를 돌파하면 매수, 20일 저가를 이탈하면 매도. 추세추종.",
                 "signal": sig_breakout},
}


# ---------- 백테스트 (롱온리, 익일 체결) ----------
def run_backtest(df, position):
    pos = position.shift(1).fillna(0)
    ret = df.close.pct_change().fillna(0)
    equity = (1 + pos * ret).cumprod()
    bh = (1 + ret).cumprod()
    dd = equity / equity.cummax() - 1
    c, p = df.close.values, pos.values
    trades, entry = [], None
    for i in range(len(p)):
        if entry is None and p[i] == 1:
            entry = i
        elif entry is not None and p[i] == 0:
            trades.append(c[i] / c[entry] - 1); entry = None
    if entry is not None:
        trades.append(c[-1] / c[entry] - 1)
    wins = [t for t in trades if t > 0]
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "bh_return": float(bh.iloc[-1] - 1),
        "mdd": float(dd.min()),
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
    }


def _epoch(index):
    """DatetimeIndex → UTC 초 (차트의 openTime//1000 과 정렬).
    pandas 2.x 는 to_datetime(unit='ms') 를 datetime64[ms] 로 만들 수 있어 asi8 해상도가
    제각각이다. datetime64[s] 로 캐스팅해 해상도와 무관하게 '초' 정수를 얻는다."""
    return [int(x) for x in index.values.astype("datetime64[s]").astype("int64")]


def current_condition(df, key):
    """지금 이 전략의 '진입 근거' — 현재 지표 상태를 사람 말로."""
    strat = STRATEGY_LIBRARY.get(key, {})
    if strat.get("custom"):
        e = strat["config"]["entry"]
        pos = strat["signal"](df)
        state = "매수/보유 신호 켜짐" if int(pos.iloc[-1]) == 1 else "관망 (신호 꺼짐)"
        return f"진입 규칙 [{e['a']} {e['op']} {e['b']}] → 현재 {state}"
    c = df.close
    if key == "sma_cross":
        f, s = sma(c, 10).iloc[-1], sma(c, 30).iloc[-1]
        up = f > s
        return f"SMA10 {f:,.0f} {'>' if up else '<'} SMA30 {s:,.0f} → " + ("골든크로스 상태 (매수 신호 유지)" if up else "데드크로스 상태 (매수 조건 아님)")
    if key == "rsi":
        r = rsi(c, 10).iloc[-1]
        up = c.iloc[-1] > sma(c, 100).iloc[-1]
        trend = "상승추세(SMA100 위)" if up else "하락추세(SMA100 아래 → 진입 보류)"
        if r < 40 and up:
            st = "과매도 눌림목 → 매수 후보"
        elif r > 70:
            st = "과매수 → 청산"
        else:
            st = "관망"
        return f"RSI {r:.0f}, {trend} → {st}"
    if key == "macd":
        m = (ema(c, 12) - ema(c, 26)); sg = ema(m, 9)
        up = m.iloc[-1] > sg.iloc[-1]
        return f"MACD {m.iloc[-1]:,.1f} {'>' if up else '<'} 시그널 {sg.iloc[-1]:,.1f} → " + ("상향 (매수 유지)" if up else "하향 (매수 조건 아님)")
    if key == "bollinger":
        ma = sma(c, 20).iloc[-1]; sd = c.rolling(20).std().iloc[-1]; low = ma - 2 * sd; p = c.iloc[-1]
        return f"현재가 {p:,.0f} vs 하단밴드 {low:,.0f} → " + ("하단 이탈 (매수 신호)" if p < low else "밴드 안 (관망)")
    if key == "breakout":
        hi = c.rolling(20).max().shift(1).iloc[-1]; p = c.iloc[-1]
        return f"현재가 {p:,.0f} vs 20일 고가 {hi:,.0f} → " + ("돌파 (매수 신호)" if p > hi else "돌파 전 (관망)")
    return ""


# ---------- 타이밍: 현재 신호 + 진입 근거 + 손절/익절 기준 ----------
def compute_timing(df, strategy_key="sma_cross"):
    if strategy_key not in STRATEGY_LIBRARY:
        strategy_key = "sma_cross"
    pos = STRATEGY_LIBRARY[strategy_key]["signal"](df)
    price = float(df.close.iloc[-1])
    a = float(atr(df).iloc[-1])
    in_pos = int(pos.iloc[-1]) == 1
    sup = float(df.low.iloc[-20:].min())
    res = float(df.high.iloc[-20:].max())

    # 손절 = 최근 스윙 저점(지지) 바로 아래. 지지가 현재가 위/너무 가까우면 ATR 기반으로 대체.
    sl = sup - 0.2 * a
    if sl >= price - 0.5 * a:
        sl = price - 1.5 * a
    risk = price - sl
    tp = price + 2.0 * risk                       # 손익비 2:1 (손절 거리의 2배 위)

    return {
        "strategy_key": strategy_key,
        "strategy_name": STRATEGY_LIBRARY[strategy_key]["name"],
        "in_position": in_pos,
        "signal": "매수/보유 구간" if in_pos else "관망/현금 구간",
        "entry_reason": current_condition(df, strategy_key),
        "sl_method": "최근 20일 스윙 저점(지지선) 바로 아래에 배치 (구조적 손절)",
        "tp_method": "손익비 2:1 — 손절까지 거리의 2배 위 (위쪽 저항은 참고: 저항 부근에서 일부 익절도 고려)",
        "price": price,
        "entry": price,
        "stop_loss": sl,
        "take_profit": tp,
        "sl_pct": (sl / price - 1) * 100,
        "tp_pct": (tp / price - 1) * 100,
        "rr": (tp - price) / risk if risk > 0 else 0.0,
        "support": sup,
        "resistance": res,
        "atr": a,
    }


# ---------- 차트 오버레이 (전략별 지표선) ----------
def indicator_overlays(df, key):
    t = _epoch(df.index)

    def ser(values):
        return [{"time": t[i], "value": float(values.iloc[i])}
                for i in range(len(values)) if pd.notna(values.iloc[i])]

    if key == "sma_cross":
        return [{"name": "SMA10", "color": "#f0b90b", "data": ser(sma(df.close, 10))},
                {"name": "SMA30", "color": "#8b5cf6", "data": ser(sma(df.close, 30))}]
    if key == "bollinger":
        ma, sd = sma(df.close, 20), df.close.rolling(20).std()
        return [{"name": "상단", "color": "#6b7280", "data": ser(ma + 2 * sd)},
                {"name": "중심(SMA20)", "color": "#f0b90b", "data": ser(ma)},
                {"name": "하단", "color": "#6b7280", "data": ser(ma - 2 * sd)}]
    if key == "breakout":
        return [{"name": "20일 고가", "color": "#26a69a", "data": ser(df.close.rolling(20).max())},
                {"name": "20일 저가", "color": "#ef5350", "data": ser(df.close.rolling(20).min())}]
    if key == "rsi":
        # 트렌드 필터(SMA100)를 오버레이로 보여줌 (RSI 자체는 별도 오실레이터)
        return [{"name": "SMA100 (추세 필터)", "color": "#8b5cf6", "data": ser(sma(df.close, 100))}]
    # macd 는 별도 오실레이터 → 가격 오버레이 없음 (마커로 신호 표시)
    return []


# ---------- 선택형 지표 오버레이 (게임/차트에서 버튼으로 표시) ----------
PRICE_INDICATORS = ["SMA 10", "SMA 20", "SMA 30", "SMA 50", "SMA 100", "볼린저밴드", "돈치안(20)"]
PANE_INDICATORS = ["RSI", "MACD"]                      # 가격 아래 별도 창(오실레이터)
AVAILABLE_INDICATORS = PRICE_INDICATORS + PANE_INDICATORS


def default_indicators(strategy_key):
    """전략에 어울리는 기본 지표 (게임/차트 시작 시 미리 선택)."""
    return {
        "sma_cross": ["SMA 10", "SMA 30"],
        "rsi": ["RSI", "SMA 100"],
        "macd": ["MACD"],
        "bollinger": ["볼린저밴드"],
        "breakout": ["돈치안(20)"],
    }.get(strategy_key, ["SMA 20"])


def pane_indicators(df, names):
    """오실레이터(RSI/MACD) → 별도 창 데이터."""
    t = _epoch(df.index)

    def ser(v):
        return [{"time": t[i], "value": float(v.iloc[i])}
                for i in range(len(v)) if pd.notna(v.iloc[i])]

    panes = []
    if "RSI" in names:
        panes.append({"id": "rsi", "lines": [
            {"color": "#f0b90b", "data": ser(rsi(df.close, 14))},
            {"color": "#6b7280", "data": ser(pd.Series(70.0, index=df.index))},
            {"color": "#6b7280", "data": ser(pd.Series(30.0, index=df.index))},
        ]})
    if "MACD" in names:
        m = ema(df.close, 12) - ema(df.close, 26)
        panes.append({"id": "macd", "lines": [
            {"color": "#3b82f6", "data": ser(m)},
            {"color": "#ef5350", "data": ser(ema(m, 9))},
        ]})
    return panes


def overlay_lines(df, names):
    """선택된 지표 이름 목록 → lightweight-charts 라인 데이터 목록."""
    t = _epoch(df.index)

    def ser(v):
        return [{"time": t[i], "value": float(v.iloc[i])}
                for i in range(len(v)) if pd.notna(v.iloc[i])]

    palette = {"SMA 10": "#f0b90b", "SMA 20": "#f7931a", "SMA 30": "#8b5cf6",
               "SMA 50": "#3b82f6", "SMA 100": "#ec4899"}
    out = []
    for n in names:
        if n in palette:
            out.append({"name": n, "color": palette[n], "data": ser(sma(df.close, int(n.split()[1])))})
        elif n == "볼린저밴드":
            ma, sd = sma(df.close, 20), df.close.rolling(20).std()
            out.append({"name": "BB 상단", "color": "#6b7280", "data": ser(ma + 2 * sd)})
            out.append({"name": "BB 중심", "color": "#f0b90b", "data": ser(ma)})
            out.append({"name": "BB 하단", "color": "#6b7280", "data": ser(ma - 2 * sd)})
        elif n == "돈치안(20)":
            out.append({"name": "20일 고가", "color": "#26a69a", "data": ser(df.close.rolling(20).max())})
            out.append({"name": "20일 저가", "color": "#ef5350", "data": ser(df.close.rolling(20).min())})
    return out


# ---------- 차트 마커 (과거 매수/매도 신호 지점) ----------
def signal_markers(df, key, max_markers=80):
    pos = STRATEGY_LIBRARY[key]["signal"](df)
    chg = pos.diff().fillna(0)
    t = _epoch(df.index)
    marks = []
    for i in range(len(pos)):
        if chg.iloc[i] == 1:
            marks.append({"time": t[i], "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "매수"})
        elif chg.iloc[i] == -1:
            marks.append({"time": t[i], "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "매도"})
    return marks[-max_markers:]


# ============================================================================
# 범용 시그널 빌더 — 리서치로 발굴한 전략(config)을 실제로 백테스트하기 위한 엔진
# ============================================================================
def _series(df, ref):
    """series 참조 문자열 → 시계열. 예: close, sma:10, ema:20, rsi:14, macd,
    macd_signal, bb_upper/bb_mid/bb_lower, donchian_high:20, donchian_low:20, const:30."""
    if ref == "close":
        return df.close
    if ref.startswith("const:"):
        return pd.Series(float(ref.split(":", 1)[1]), index=df.index)
    if ref.startswith("sma:"):
        return sma(df.close, int(ref.split(":", 1)[1]))
    if ref.startswith("ema:"):
        return ema(df.close, int(ref.split(":", 1)[1]))
    if ref.startswith("rsi:"):
        return rsi(df.close, int(ref.split(":", 1)[1]))
    if ref == "macd":
        return ema(df.close, 12) - ema(df.close, 26)
    if ref == "macd_signal":
        return ema(ema(df.close, 12) - ema(df.close, 26), 9)
    if ref in ("bb_upper", "bb_mid", "bb_lower"):
        ma, sd = sma(df.close, 20), df.close.rolling(20).std()
        return {"bb_upper": ma + 2 * sd, "bb_mid": ma, "bb_lower": ma - 2 * sd}[ref]
    if ref.startswith("donchian_high:"):
        return df.close.rolling(int(ref.split(":", 1)[1])).max().shift(1)
    if ref.startswith("donchian_low:"):
        return df.close.rolling(int(ref.split(":", 1)[1])).min().shift(1)
    raise ValueError(f"unknown series: {ref}")


def _cond(df, rule):
    a, b = _series(df, rule["a"]), _series(df, rule["b"])
    op = rule["op"]
    if op == "cross_above":
        return (a > b) & (a.shift() <= b.shift())
    if op == "cross_below":
        return (a < b) & (a.shift() >= b.shift())
    if op == "above":
        return a > b
    if op == "below":
        return a < b
    raise ValueError(f"unknown op: {op}")


SERIES_VOCAB = ["close", "sma:N", "ema:N", "rsi:N", "macd", "macd_signal",
                "bb_upper", "bb_mid", "bb_lower", "donchian_high:N", "donchian_low:N", "const:X"]
OP_VOCAB = ["cross_above", "cross_below", "above", "below"]


def build_signal(df, cfg):
    """전략 config({entry:{a,op,b}, exit:{a,op,b}}) → position 시계열."""
    entry = _cond(df, cfg["entry"]).fillna(False)
    exit = _cond(df, cfg["exit"]).fillna(False)
    return _hold(entry, exit)


# ============================================================================
# 커스텀(리서치로 편입한) 전략 등록 · 로딩
# ============================================================================
def register_custom(cfg):
    """config 를 STRATEGY_LIBRARY 에 등록 (즉시 학습/게임/차트에서 사용 가능)."""
    key = cfg["key"]
    STRATEGY_LIBRARY[key] = {
        "name": cfg["name"],
        "desc": cfg.get("desc", ""),
        "signal": (lambda c: (lambda df: build_signal(df, c)))(cfg),
        "custom": True,
        "config": cfg,
        "indicators": cfg.get("indicators", []),
    }
    # 커스텀 전략의 기본 지표를 default_indicators 로도 노출
    _CUSTOM_DEFAULT_INDICATORS[key] = cfg.get("indicators", [])


_CUSTOM_DEFAULT_INDICATORS = {}
_orig_default_indicators = default_indicators


def default_indicators(strategy_key):  # noqa: F811  (커스텀 포함하도록 확장)
    if strategy_key in _CUSTOM_DEFAULT_INDICATORS:
        return _CUSTOM_DEFAULT_INDICATORS[strategy_key] or ["SMA 20"]
    return _orig_default_indicators(strategy_key)


def _load_saved_custom():
    """저장된 커스텀(편입) 전략을 STRATEGY_LIBRARY 에 로드 (앱 시작 시)."""
    import json
    import pathlib
    f = pathlib.Path(__file__).parent / "custom_strategies.json"
    if f.exists():
        try:
            for cfg in json.loads(f.read_text()).values():
                register_custom(cfg)
        except Exception:
            pass


_load_saved_custom()
