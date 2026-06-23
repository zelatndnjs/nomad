import dotenv

dotenv.load_dotenv()

import asyncio
import random

import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    RunContextWrapper,
    function_tool,
    AgentHooks,
    Tool,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel

st.set_page_config(page_title="Restaurant Bot", page_icon="🍽️")
st.title("🍽️ 노마드 비스트로 봇")
st.caption("메뉴 · 주문 · 예약을 도와드려요. Triage 에이전트가 알맞은 전문가에게 연결합니다.")


# =============================================================================
# CONTEXT
# =============================================================================


class CustomerContext(BaseModel):

    name: str = "손님"


customer_ctx = CustomerContext(name="손님")


# =============================================================================
# MENU DATA
# =============================================================================

MENU = {
    "스타터": [
        {
            "name": "토마토 브루스케타",
            "price": 9000,
            "ingredients": ["바게트", "토마토", "바질", "올리브오일", "마늘"],
            "allergens": ["글루텐"],
            "tags": ["채식"],
        },
        {
            "name": "버섯 크림수프",
            "price": 11000,
            "ingredients": ["양송이버섯", "생크림", "버터", "양파"],
            "allergens": ["유제품"],
            "tags": ["채식"],
        },
    ],
    "메인": [
        {
            "name": "트러플 마르게리타 피자",
            "price": 21000,
            "ingredients": ["밀가루 도우", "모짜렐라", "토마토소스", "바질", "트러플오일"],
            "allergens": ["글루텐", "유제품"],
            "tags": ["채식"],
        },
        {
            "name": "그릴드 연어 스테이크",
            "price": 29000,
            "ingredients": ["연어", "아스파라거스", "레몬버터소스"],
            "allergens": ["생선", "유제품"],
            "tags": [],
        },
        {
            "name": "트러플 크림 파스타",
            "price": 23000,
            "ingredients": ["페투치네", "생크림", "파마산", "트러플"],
            "allergens": ["글루텐", "유제품", "달걀"],
            "tags": ["채식"],
        },
        {
            "name": "비욘드 베지 버거",
            "price": 19000,
            "ingredients": ["식물성 패티", "번", "양상추", "토마토", "비건 마요"],
            "allergens": ["글루텐", "대두"],
            "tags": ["채식", "비건"],
        },
    ],
    "디저트": [
        {
            "name": "티라미수",
            "price": 9000,
            "ingredients": ["마스카포네", "에스프레소", "레이디핑거", "코코아"],
            "allergens": ["유제품", "달걀", "글루텐"],
            "tags": ["채식"],
        },
        {
            "name": "비건 초코 무스",
            "price": 8500,
            "ingredients": ["아보카도", "코코아", "메이플시럽", "코코넛크림"],
            "allergens": [],
            "tags": ["채식", "비건"],
        },
    ],
}


def _price_map():
    return {dish["name"]: dish["price"] for dishes in MENU.values() for dish in dishes}


# =============================================================================
# TOOL USAGE HOOKS (sidebar logging)
# =============================================================================


class ToolLoggingHooks(AgentHooks):

    async def on_tool_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
    ):
        with st.sidebar:
            st.caption(f"🔧 {agent.name} → `{tool.name}` 실행 중...")

    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        result: str,
    ):
        with st.sidebar:
            st.caption(f"✅ `{tool.name}` 완료")


# =============================================================================
# MENU TOOLS
# =============================================================================


@function_tool
def get_full_menu() -> str:
    """전체 메뉴를 카테고리, 가격, 재료, 알레르기 정보와 함께 반환합니다."""
    lines = []
    for category, dishes in MENU.items():
        lines.append(f"【{category}】")
        for dish in dishes:
            tags = f" ({', '.join(dish['tags'])})" if dish["tags"] else ""
            allergens = ", ".join(dish["allergens"]) if dish["allergens"] else "없음"
            lines.append(
                f"- {dish['name']}{tags} — {dish['price']:,}원 "
                f"| 재료: {', '.join(dish['ingredients'])} | 알레르기 유발: {allergens}"
            )
    return "\n".join(lines)


@function_tool
def find_dishes_by_tag(tag: str) -> str:
    """식단 태그(예: '채식', '비건')로 메뉴를 검색합니다.

    Args:
        tag: 검색할 식단 태그
    """
    matches = []
    for dishes in MENU.values():
        for dish in dishes:
            if any(tag in t for t in dish["tags"]):
                matches.append(f"- {dish['name']} ({dish['price']:,}원)")
    if not matches:
        return f"'{tag}' 에 해당하는 메뉴를 찾지 못했어요."
    return f"'{tag}' 메뉴:\n" + "\n".join(matches)


@function_tool
def check_allergens(dish_name: str) -> str:
    """특정 요리의 알레르기 유발 성분과 재료를 확인합니다.

    Args:
        dish_name: 확인할 요리 이름
    """
    for dishes in MENU.values():
        for dish in dishes:
            if dish_name in dish["name"] or dish["name"] in dish_name:
                allergens = ", ".join(dish["allergens"]) if dish["allergens"] else "없음"
                return (
                    f"{dish['name']} 의 알레르기 유발 성분: {allergens}\n"
                    f"재료: {', '.join(dish['ingredients'])}"
                )
    return f"'{dish_name}' 메뉴를 찾지 못했어요. 메뉴 이름을 다시 확인해 주세요."


# =============================================================================
# ORDER TOOLS
# =============================================================================


@function_tool
def place_order(items: list[str]) -> str:
    """고객의 주문을 받아 주문 번호와 합계를 반환합니다.

    Args:
        items: 주문할 메뉴 이름 목록
    """
    price_map = _price_map()
    order_lines = []
    not_found = []
    total = 0
    for item in items:
        matched = None
        for name, price in price_map.items():
            if item in name or name in item:
                matched = (name, price)
                break
        if matched:
            order_lines.append(f"- {matched[0]}: {matched[1]:,}원")
            total += matched[1]
        else:
            not_found.append(item)

    order_id = f"ORD-{random.randint(1000, 9999)}"
    result = f"🧾 주문 번호: {order_id}\n" + "\n".join(order_lines)
    result += f"\n합계: {total:,}원"
    if not_found:
        result += f"\n⚠️ 메뉴에서 찾지 못한 항목: {', '.join(not_found)}"
    return result


# =============================================================================
# RESERVATION TOOLS
# =============================================================================


@function_tool
def check_availability(date: str, time: str, party_size: int) -> str:
    """요청한 날짜/시간/인원에 예약 가능한지 확인합니다.

    Args:
        date: 희망 날짜 (예: 2026-06-25)
        time: 희망 시간 (예: 19:00)
        party_size: 인원수
    """
    available = random.random() > 0.25
    if available:
        return f"{date} {time}, {party_size}명 예약 가능합니다!"
    alt = "18:00" if time > "18:00" else "20:30"
    return f"{date} {time} 는 만석이에요. {alt} 또는 다른 시간은 어떠세요?"


@function_tool
def make_reservation(name: str, party_size: int, date: str, time: str) -> str:
    """테이블 예약을 생성하고 예약 확인 정보를 반환합니다.

    Args:
        name: 예약자 이름
        party_size: 인원수
        date: 예약 날짜 (예: 2026-06-25)
        time: 예약 시간 (예: 19:00)
    """
    reservation_id = f"RES-{random.randint(1000, 9999)}"
    return (
        f"✅ 예약이 확정되었습니다!\n"
        f"예약 번호: {reservation_id}\n"
        f"예약자: {name}\n"
        f"인원: {party_size}명\n"
        f"날짜: {date}\n"
        f"시간: {time}\n"
        f"방문 10분 전까지 도착 부탁드려요."
    )


# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================


def triage_instructions(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
):
    return f"""{RECOMMENDED_PROMPT_PREFIX}

    너는 '노마드 비스트로' 레스토랑의 안내(Triage) 담당이야.
    손님({wrapper.context.name})을 따뜻하게 환영하고, 무엇을 원하는지 빠르게 파악해서
    알맞은 전문 담당에게 연결(handoff)해.

    라우팅 규칙:
    - 메뉴, 재료, 알레르기, 채식/비건 등 음식 관련 질문 → Menu Agent
    - 음식을 주문하거나 주문을 확정/변경 → Order Agent
    - 테이블 예약, 예약 변경/확인 → Reservation Agent

    핸드오프 직전에 한국어로 짧게 어디로 연결하는지 안내하고 바로 해당 담당에게 넘겨.
    레스토랑과 무관한 질문에는 정중히 도와줄 수 없다고 답해.
    항상 한국어로 친근하게 답해.
    """


def menu_instructions(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
):
    return f"""{RECOMMENDED_PROMPT_PREFIX}

    너는 '노마드 비스트로'의 메뉴 전문가야.
    손님({wrapper.context.name})의 메뉴/재료/알레르기/식단(채식·비건) 질문에 답해.

    규칙:
    - 메뉴를 물으면 get_full_menu 도구로 정확한 메뉴/가격/재료/알레르기 정보를 확인해.
    - 채식, 비건 등 식단 질문은 find_dishes_by_tag 로 검색해.
    - 특정 요리의 알레르기 성분은 check_allergens 로 확인해. 절대 추측하지 마.
    - 손님이 주문을 원하면 Order Agent 로, 예약을 원하면 Reservation Agent 로 핸드오프해.
    항상 한국어로 친근하게 답해.
    """


def order_instructions(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
):
    return f"""{RECOMMENDED_PROMPT_PREFIX}

    너는 '노마드 비스트로'의 주문 담당이야.
    손님({wrapper.context.name})의 주문을 받고 확인해.

    규칙:
    - 손님이 주문할 메뉴를 말하면 항목을 확인하고 place_order 도구로 주문을 생성해.
    - 주문 후 주문 번호와 합계를 명확히 알려주고, 더 필요한 게 있는지 물어봐.
    - 메뉴/재료 질문은 Menu Agent 로, 예약은 Reservation Agent 로 핸드오프해.
    항상 한국어로 친근하게 답해.
    """


def reservation_instructions(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
):
    return f"""{RECOMMENDED_PROMPT_PREFIX}

    너는 '노마드 비스트로'의 예약 담당이야.
    손님({wrapper.context.name})의 테이블 예약을 처리해.

    규칙:
    - 예약에 필요한 정보(예약자 이름, 인원수, 날짜, 시간)를 정중히 물어봐.
      빠진 정보가 있으면 예약을 확정하기 전에 먼저 확인해.
    - 필요하면 check_availability 로 가능 여부를 확인하고, make_reservation 으로 예약을 확정해.
    - 확정 후 예약 번호와 상세 정보를 안내해.
    - 메뉴/재료 질문은 Menu Agent 로, 주문은 Order Agent 로 핸드오프해.
    항상 한국어로 친근하게 답해.
    """


# =============================================================================
# AGENTS
# =============================================================================

menu_agent = Agent(
    name="Menu Agent",
    handoff_description="메뉴, 재료, 알레르기, 채식/비건 등 음식 관련 질문 담당",
    instructions=menu_instructions,
    tools=[get_full_menu, find_dishes_by_tag, check_allergens],
    hooks=ToolLoggingHooks(),
)

order_agent = Agent(
    name="Order Agent",
    handoff_description="음식 주문 접수 및 확인 담당",
    instructions=order_instructions,
    tools=[place_order],
    hooks=ToolLoggingHooks(),
)

reservation_agent = Agent(
    name="Reservation Agent",
    handoff_description="테이블 예약 처리 담당",
    instructions=reservation_instructions,
    tools=[check_availability, make_reservation],
    hooks=ToolLoggingHooks(),
)

triage_agent = Agent(
    name="Triage Agent",
    handoff_description="요청을 파악해 알맞은 전문 담당으로 연결",
    instructions=triage_instructions,
)

# 에이전트 간 상호 핸드오프 연결 (생성 후 설정해 순환 참조를 피한다).
# 어느 담당과 대화 중이어도 다른 전문가에게 바로 넘어갈 수 있다.
triage_agent.handoffs = [menu_agent, order_agent, reservation_agent]
menu_agent.handoffs = [triage_agent, order_agent, reservation_agent]
order_agent.handoffs = [triage_agent, menu_agent, reservation_agent]
reservation_agent.handoffs = [triage_agent, menu_agent, order_agent]


# 핸드오프 발생 시 UI 에 보여줄 한국어 안내 문구.
HANDOFF_NOTICE = {
    "Menu Agent": "🍽️ 메뉴 전문가에게 연결합니다...",
    "Order Agent": "📝 주문 담당에게 연결합니다...",
    "Reservation Agent": "📅 예약 담당에게 연결합니다...",
    "Triage Agent": "🧭 안내 담당에게 다시 연결합니다...",
}


# =============================================================================
# SESSION / STATE
# =============================================================================

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "restaurant-memory.db",
    )
session = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent


async def paint_history():
    messages = await session.get_items()
    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message.get("type") == "message":
                        st.write(message["content"][0]["text"].replace("$", r"\$"))


asyncio.run(paint_history())


async def run_agent(message):

    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""

        st.session_state["text_placeholder"] = text_placeholder

        stream = Runner.run_streamed(
            st.session_state["agent"],
            message,
            session=session,
            context=customer_ctx,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", r"\$"))

            elif event.type == "agent_updated_stream_event":

                current_name = st.session_state["agent"].name
                new_name = event.new_agent.name

                if current_name != new_name:

                    notice = HANDOFF_NOTICE.get(
                        new_name, f"🔀 {new_name} 에게 연결합니다..."
                    )
                    st.info(notice)

                    st.session_state["agent"] = event.new_agent

                    text_placeholder = st.empty()
                    st.session_state["text_placeholder"] = text_placeholder
                    response = ""


message = st.chat_input("무엇을 도와드릴까요? (메뉴 · 주문 · 예약)")

if message:
    with st.chat_message("human"):
        st.write(message)
    asyncio.run(run_agent(message))


with st.sidebar:
    st.header("🍽️ 노마드 비스트로")
    st.caption(f"현재 담당: **{st.session_state['agent'].name}**")
    st.divider()
    st.subheader("에이전트 구성")
    st.markdown(
        "- 🧭 **Triage** — 요청 파악 후 연결\n"
        "- 🍽️ **Menu** — 메뉴 · 재료 · 알레르기\n"
        "- 📝 **Order** — 주문 접수 · 확인\n"
        "- 📅 **Reservation** — 테이블 예약"
    )
    st.divider()
    if st.button("대화 초기화"):
        asyncio.run(session.clear_session())
        st.session_state["agent"] = triage_agent
        st.rerun()
