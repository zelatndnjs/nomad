"""도구(Tool) + 학습/피드백 헬퍼.

- TOOLS: analyze_timing · market_snapshot · web_search (supervisor 의 Researcher 가 사용)
- learn(): 전략 규칙 + 백테스트 + 코칭 (전략 학습 탭)
- trade_feedback(): 연습 게임 결과 코치 피드백
"""
import dotenv
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from .data import fetch_df, ticker_24h
from .strategies import STRATEGY_LIBRARY, run_backtest, compute_timing

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ======================= 도구 =======================
@tool
def analyze_timing(symbol: str, strategy_key: str = "sma_cross") -> str:
    """지금 시점의 매수/매도 신호와 진입가·손절가·익절가를 실제 코인 데이터로 계산한다.
    Args:
        symbol: 코인 심볼 (예: BTCUSDT)
        strategy_key: sma_cross/rsi/macd/bollinger/breakout
    """
    try:
        t = compute_timing(fetch_df(symbol, "1d", 200), strategy_key)
        return (f"[{symbol} · {t['strategy_name']}] 타이밍 (학습용, 투자 조언 아님)\n"
                f"- 현재가 {t['price']:,.2f} / 신호 {t['signal']}\n"
                f"- 진입 {t['entry']:,.2f} / 손절 {t['stop_loss']:,.2f} ({t['sl_pct']:+.1f}%) / "
                f"익절 {t['take_profit']:,.2f} ({t['tp_pct']:+.1f}%)\n"
                f"- 손익비 {t['rr']:.2f}:1 / 지지 {t['support']:,.2f} 저항 {t['resistance']:,.2f}\n"
                f"- 진입 근거: {t['entry_reason']}")
    except Exception as e:
        return f"타이밍 분석 실패: {e}"


@tool
def market_snapshot(symbol: str) -> str:
    """코인의 현재가·24시간 변동률·고저를 반환한다.
    Args:
        symbol: 코인 심볼 (예: BTCUSDT)
    """
    try:
        d = ticker_24h(symbol)
        return f"{symbol} 현재가 {d['price']:,.2f} / 24h {d['change_pct']:+.2f}% / 고 {d['high']:,.2f} 저 {d['low']:,.2f}"
    except Exception as e:
        return f"시세 조회 실패: {e}"


@tool
def web_search(query: str) -> str:
    """인터넷에서 매매 전략·시장 정보를 검색한다.
    Args:
        query: 검색어
    """
    try:
        from ddgs import DDGS
        with DDGS() as d:
            res = list(d.text(query, max_results=3))
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')[:160]}" for r in res) or "검색 결과 없음"
    except Exception as e:
        return f"웹 검색 사용 불가: {e}"


TOOLS = [analyze_timing, market_snapshot, web_search]


# ======================= 학습 / 피드백 =======================
def learn(strategy_key: str, symbol: str = "BTCUSDT"):
    """전략 학습: 규칙 + 백테스트(긴 히스토리) + 코칭."""
    if strategy_key not in STRATEGY_LIBRARY:
        strategy_key = "sma_cross"
    strat = STRATEGY_LIBRARY[strategy_key]
    df = fetch_df(symbol, "1d", 1500)   # ~4년으로 정확한 백테스트
    bt = run_backtest(df, strat["signal"](df))
    rules = llm.invoke([SystemMessage(content="트레이딩 교육 튜터"),
        HumanMessage(content=f"전략 '{strat['name']}' ({strat['desc']}) 을 진입규칙/청산규칙/핵심원리/주의점으로 한국어로 간결히 가르쳐. 학습용.")]).content
    coaching = llm.invoke([SystemMessage(content="트레이딩 교육 튜터"),
        HumanMessage(content=f"{symbol} 에 '{strat['name']}' 백테스트 결과(전략 {bt['total_return']:.1%}, 보유 {bt['bh_return']:.1%}, MDD {bt['mdd']:.1%}, {bt['n_trades']}회, 승률 {bt['win_rate']:.0%})를 학습 관점으로 짧게 해석·코칭해.")]).content
    return {"strategy_key": strategy_key, "strategy_name": strat["name"],
            "symbol": symbol, "rules": rules, "backtest": bt, "coaching": coaching}


def trade_feedback(info: dict):
    """연습 게임 결과에 대한 트레이딩 코치 피드백."""
    prompt = (
        f"학생이 트레이딩 연습 게임을 했어.\n"
        f"진입가 {info['entry']:.2f}, 익절 {info['tp']:.2f}({info['tp_pct']:+.1f}%), "
        f"손절 {info['sl']:.2f}({info['sl_pct']:+.1f}%), 손익비 {info['rr']:.2f}:1.\n"
        f"결과: {info['outcome_kr']}, 실제 수익률 {info['pnl']:+.1f}% "
        f"(끝까지 보유했다면 {info['bh']:+.1f}%).\n\n"
        f"트레이딩 학습 코치로서 한국어로 3~4줄:\n"
        f"1) 이 진입/손절/익절 설정(특히 손익비)이 적절했는지\n"
        f"2) 이 결과에서 배울 점\n"
        f"3) 다음엔 무엇을 다르게 할지\n"
        f"학습용이고 격려하는 톤으로."
    )
    return llm.invoke([SystemMessage(content="트레이딩 학습 코치"), HumanMessage(content=prompt)]).content
