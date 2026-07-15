"""고급 패턴 — 멀티 에이전트 (Supervisor 아키텍처).

Supervisor 가 사용자 입력을 보고 전문 에이전트로 라우팅한다:

    Supervisor ─┬─ Tutor       (전략·개념 설명)
                ├─ Researcher  (실데이터 도구: 타이밍/시세/검색)
                └─ Quiz        (트레이딩 퀴즈 출제·채점)

메모리(MemorySaver)로 대화 맥락을 유지 → 퀴즈 출제 후 다음 턴에서 답을 채점할 수 있다.
"""
from typing import TypedDict, Annotated, Literal

import dotenv
dotenv.load_dotenv()

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .agent import TOOLS          # analyze_timing · market_snapshot · web_search 재사용
from .strategies import STRATEGY_LIBRARY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_tools = llm.bind_tools(TOOLS)
STRAT_LIST = ", ".join(v["name"] for v in STRATEGY_LIBRARY.values())


class SuperState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


class Route(BaseModel):
    next: Literal["tutor", "researcher", "quiz"] = Field(
        description="어느 전문 에이전트가 답할지"
    )


SUP_SYS = SystemMessage(content=(
    "너는 트레이딩 학습 플랫폼의 Supervisor 야. 사용자 메시지를 보고 알맞은 전문 에이전트를 하나 골라.\n"
    "- tutor: 매매 전략·개념 설명, '알려줘/설명해/뭐야/어떻게' 류 학습 질문\n"
    "- researcher: 지금 시세·타이밍·손절/익절 등 '실제 데이터'가 필요한 질문\n"
    "- quiz: 사용자가 퀴즈를 원하거나, 직전에 낸 퀴즈에 답을 제출했을 때\n"
))


# ---------- Supervisor ----------
def supervisor(state: SuperState):
    r = llm.with_structured_output(Route).invoke([SUP_SYS] + state["messages"][-6:])
    return {"next": r.next}


# ---------- 전문 에이전트들 ----------
def tutor(state: SuperState):
    sys = SystemMessage(content=(
        f"너는 트레이딩 학습 튜터야. 매매 전략과 개념을 쉽고 명확하게 한국어로 설명해. "
        f"내장 전략: {STRAT_LIST}. 진입/청산 규칙, 손절·익절, 손익비 개념을 학습 관점으로 가르쳐. "
        f"투자 조언이 아니라 학습임을 밝혀."))
    return {"messages": [llm.invoke([sys] + state["messages"])]}


RES_SYS = SystemMessage(content=(
    "너는 리서처야. analyze_timing / market_snapshot / web_search 도구로 실제 데이터를 가져와 "
    "질문에 답하고 근거(변동성·지지/저항·손익비)를 설명해. 항상 한국어. 학습용이며 투자 조언이 아님."))


def researcher(state: SuperState):
    return {"messages": [llm_tools.invoke([RES_SYS] + state["messages"])]}


def quiz(state: SuperState):
    sys = SystemMessage(content=(
        f"너는 퀴즈 마스터야. 트레이딩 전략·개념 4지선다 퀴즈를 낸다.\n"
        f"- 사용자가 방금 퀴즈에 답했으면: 정답 여부 + 해설을 알려주고 한 줄 격려.\n"
        f"- 아니면: 새 문제 1개를 (보기 A~D 포함) 낸다.\n"
        f"주제: {STRAT_LIST}, 손절/익절/손익비/추세추종·평균회귀 등. 한국어, 학습용."))
    return {"messages": [llm.invoke([sys] + state["messages"])]}


def _route(state: SuperState) -> str:
    return state["next"]


def _build():
    g = StateGraph(SuperState)
    g.add_node("supervisor", supervisor)
    g.add_node("tutor", tutor)
    g.add_node("researcher", researcher)
    g.add_node("quiz", quiz)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "supervisor")
    # Supervisor → 전문 에이전트 (Conditional Edge)
    g.add_conditional_edges("supervisor", _route,
                            {"tutor": "tutor", "researcher": "researcher", "quiz": "quiz"})
    g.add_edge("tutor", END)
    g.add_edge("quiz", END)
    # Researcher 는 도구를 쓸 수 있음 (tools_condition 루프)
    g.add_conditional_edges("researcher", tools_condition)
    g.add_edge("tools", "researcher")
    return g.compile(checkpointer=MemorySaver())


SUPERVISOR_APP = _build()


def run(message: str, thread_id: str = "st"):
    """멀티에이전트 실행 → 답변 + 어느 전문가가 답했는지."""
    cfg = {"configurable": {"thread_id": thread_id}}
    out = SUPERVISOR_APP.invoke({"messages": [HumanMessage(content=message)]}, cfg)
    return {"reply": out["messages"][-1].content, "agent": out.get("next", "")}
