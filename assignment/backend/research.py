"""전략 리서치 자동화 — AI가 새 전략을 발굴 → 실제 백테스트 → AI 판정(유효/폐기)
→ 사용자가 편입(adopt)하면 학습/게임/차트에 즉시 반영, 폐기(discard)는 기록만. 내역 저장."""
import json
import statistics as st
from datetime import datetime
from pathlib import Path
from typing import Literal

import dotenv
dotenv.load_dotenv()

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .data import fetch_df, TOP_COINS
from .strategies import (
    STRATEGY_LIBRARY, run_backtest, build_signal, register_custom,
    SERIES_VOCAB, OP_VOCAB, AVAILABLE_INDICATORS,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

CUSTOM_FILE = Path(__file__).parent / "custom_strategies.json"
LOG_FILE = Path(__file__).parent / "research_log.json"


# ---------- 파일 IO ----------
def load_custom():
    return json.loads(CUSTOM_FILE.read_text()) if CUSTOM_FILE.exists() else {}


def save_custom(d):
    CUSTOM_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def load_log():
    return json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []


def _append_log(entry):
    log = load_log()
    log.insert(0, entry)
    LOG_FILE.write_text(json.dumps(log, ensure_ascii=False, indent=2))


# ---------- 1) 리서치: AI가 전략 config 생성 ----------
class Rule(BaseModel):
    a: str = Field(description="series 참조 (예: sma:5, close, rsi:14, macd)")
    op: Literal["cross_above", "cross_below", "above", "below"]
    b: str = Field(description="series 참조 (예: sma:20, const:30, macd_signal)")


class StrategyProposal(BaseModel):
    name: str = Field(description="전략 이름 (한국어, 짧게)")
    desc: str = Field(description="한 줄 설명 (한국어)")
    entry: Rule
    exit: Rule
    indicators: list[str] = Field(description="차트에 표시할 지표 이름")


def research_strategy(theme: str = ""):
    existing = ", ".join(v["name"] for v in STRATEGY_LIBRARY.values())
    prompt = f"""너는 퀀트 전략 연구원이야. 백테스트로 검증할 '새로운 매매 전략' 하나를 제안해.

시리즈(series) 참조는 다음만 사용 (N=기간 숫자, X=상수): {SERIES_VOCAB}
조건 연산: {OP_VOCAB}
entry(진입)와 exit(청산)를 각각 {{a, op, b}} 규칙으로.
indicators 는 차트 표시용, 다음 중에서 관련된 것: {AVAILABLE_INDICATORS}

{('테마/방향: ' + theme) if theme else '유명하거나 검증된 아이디어의 변형/조합으로.'}
기존 전략들과 겹치지 않게: {existing}
롱(매수) 전용. 진입은 상승 신호, 청산은 하락/반대 신호로."""
    p = llm.with_structured_output(StrategyProposal).invoke([HumanMessage(content=prompt)])
    cfg = p.model_dump()
    cfg["key"] = f"custom_{len(load_custom()) + 1}_{abs(hash(cfg['name'])) % 10000}"
    return cfg


# ---------- 2) 백테스트 (상위 5코인 · 일봉 ~4년) ----------
def evaluate(cfg):
    rows = []
    for c in TOP_COINS:
        try:
            df = fetch_df(c, "1d", 1500)
            r = run_backtest(df, build_signal(df, cfg))
        except Exception as e:
            return {"error": f"백테스트 실패 ({c}): {e}"}
        r["excess"] = r["total_return"] - r["bh_return"]
        r["rpm"] = r["total_return"] / abs(r["mdd"]) if r["mdd"] else 0.0
        r["coin"] = c
        rows.append(r)
    avg = lambda k: st.mean(x[k] for x in rows)
    summary = {"ret": avg("total_return"), "bh": avg("bh_return"), "mdd": avg("mdd"),
               "rpm": avg("rpm"), "win": avg("win_rate"), "trades": avg("n_trades"),
               "beats": sum(1 for x in rows if x["excess"] > 0)}
    return {"rows": rows, "summary": summary}


# ---------- 3) AI 판정 (유효/폐기) ----------
def is_valid(summary):
    """유효 판정 기준: 수익>0, 수익/MDD≥1.5, 매매≥1 (순수 함수, 테스트 가능)."""
    return summary["ret"] > 0 and summary["rpm"] >= 1.5 and summary["trades"] >= 1


def judge(cfg, ev):
    s = ev["summary"]
    valid = is_valid(s)
    text = llm.invoke([
        SystemMessage(content="퀀트 전략 평가자. 냉정하게 판단한다."),
        HumanMessage(content=(
            f"전략 '{cfg['name']}' ({cfg['desc']}) 백테스트(상위 5코인·일봉 4년):\n"
            f"평균수익 {s['ret']:.0%}, 매수후보유 {s['bh']:.0%}, MDD {s['mdd']:.0%}, "
            f"수익/MDD {s['rpm']:.2f}, 승률 {s['win']:.0%}, 평균 {s['trades']:.0f}회, B&H 이긴 코인 {s['beats']}/5.\n\n"
            f"판정: {'유효' if valid else '무효(폐기 권장)'} (기준: 수익>0, 수익/MDD≥1.5, 매매≥1).\n"
            f"평가자로서 한국어 3~4줄: 이 판정의 근거 + 전략의 강점/약점 + 학습 포인트."))
    ]).content
    return {"valid": valid, "verdict": "유효 ✅" if valid else "무효 ❌ (폐기 권장)", "text": text}


# ---------- 4) 편입 / 폐기 + 내역 ----------
def _log_entry(cfg, ev, decision):
    s = ev["summary"]
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": cfg["name"], "desc": cfg["desc"], "decision": decision,
            "ret": round(s["ret"], 3), "rpm": round(s["rpm"], 2),
            "win": round(s["win"], 3), "trades": round(s["trades"]),
            "key": cfg.get("key", "")}


def adopt(cfg, ev):
    """유효 전략을 편입 → 학습/게임/차트에서 즉시 사용 + 저장."""
    d = load_custom()
    d[cfg["key"]] = cfg
    save_custom(d)
    register_custom(cfg)
    _append_log(_log_entry(cfg, ev, "편입"))


def discard(cfg, ev):
    """폐기 → 기록만."""
    _append_log(_log_entry(cfg, ev, "폐기"))
