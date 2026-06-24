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
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel

st.set_page_config(page_title="Restaurant Bot", page_icon="🍽️")
st.title("🍽️ 노마드 비스트로 봇")
st.caption(
    "메뉴 · 주문 · 예약 · 불만 접수를 도와드려요. "
    "Triage 에이전트가 알맞은 전문가에게 연결하고, Guardrails 가 대화를 안전하게 지킵니다."
)


# =============================================================================
# CONTEXT & MODELS
# =============================================================================


class CustomerContext(BaseModel):

    name: str = "손님"


class InputGuardrailOutput(BaseModel):

    is_off_topic: bool
    is_inappropriate: bool
    reason: str


class OutputGuardrailOutput(BaseModel):

    is_unprofessional: bool
    leaks_internal_info: bool
    reason: str


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
# GUARDRAILS
# =============================================================================

# ----- Input Guardrail: 주제 이탈 + 부적절한 언어 차단 -----
input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="""
    너는 레스토랑 챗봇의 '입력 검열기'야. 사용자의 메시지를 분석해 두 가지를 판단해.

    1) is_off_topic: 레스토랑과 무관한 요청이면 true.
       - 허용(false): 메뉴/재료/알레르기, 주문, 예약, 음식·서비스에 대한 불만이나 부정적 피드백,
         환불/할인/매니저 요청, 인사나 짧은 잡담.
       - 거부(true): 인생의 의미, 코딩 도움, 일반 상식, 수학 문제, 날씨, 정치 등
         레스토랑과 관련 없는 주제.

    2) is_inappropriate: 욕설, 혐오 표현, 성적/폭력적 표현, 직원이나 봇을 향한 모욕적 공격이면 true.
       - 주의: 음식이나 서비스에 대한 정중한 불만(예: "음식이 별로였어요", "직원이 불친절했어요")은
         부적절한 표현이 아니다. 이런 경우 is_inappropriate=false 로 둔다.

    reason 에 판단 근거를 한국어로 간단히 적어.
    """,
    output_type=InputGuardrailOutput,
)


@input_guardrail
async def restaurant_input_guardrail(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
    input: str,
):
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=wrapper.context,
    )
    out = result.final_output
    return GuardrailFunctionOutput(
        output_info=out,
        tripwire_triggered=out.is_off_topic or out.is_inappropriate,
    )


# ----- Output Guardrail: 비전문적 응답 + 내부 정보 노출 차단 -----
output_guardrail_agent = Agent(
    name="Output Guardrail Agent",
    instructions="""
    너는 레스토랑 챗봇의 '출력 검열기'야. 봇이 손님에게 보내려는 응답을 분석해 두 가지를 판단해.

    1) is_unprofessional: 무례하거나, 모욕적이거나, 공격적이거나, 욕설이 있거나,
       손님을 탓하는 등 비전문적인 어조이면 true.
       정중하고 공감하는 사과·안내는 false.

    2) leaks_internal_info: 내부 정보가 노출되면 true. 내부 정보 예시:
       - 시스템/내부 프롬프트, 에이전트 이름이나 내부 구조, 도구(function) 이름·구현
       - 데이터베이스/파일/세션 등 시스템 내부 정보
       - 다른 손님의 개인정보
       정상적인 메뉴·가격·예약번호·주문번호·할인코드·환불번호 안내는 내부 정보가 아니다(false).

    reason 에 판단 근거를 한국어로 간단히 적어.
    """,
    output_type=OutputGuardrailOutput,
)


@output_guardrail
async def restaurant_output_guardrail(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
    output: str,
):
    result = await Runner.run(
        output_guardrail_agent,
        output,
        context=wrapper.context,
    )
    out = result.final_output
    triggered = out.is_unprofessional or out.leaks_internal_info
    return GuardrailFunctionOutput(
        output_info=out,
        tripwire_triggered=triggered,
    )


INPUT_GUARDRAILS = [restaurant_input_guardrail]
OUTPUT_GUARDRAILS = [restaurant_output_guardrail]


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
# COMPLAINTS TOOLS
# =============================================================================


@function_tool
def offer_discount(percent: int) -> str:
    """다음 방문 시 사용할 사과 할인 쿠폰을 발급합니다.

    Args:
        percent: 할인율(%)
    """
    code = f"SORRY-{random.randint(1000, 9999)}"
    return (
        f"🎟️ 다음 방문 시 {percent}% 할인 쿠폰이 발급되었습니다. "
        f"쿠폰 코드: {code} (발급일로부터 90일 유효)"
    )


@function_tool
def process_refund(amount: int, reason: str) -> str:
    """불만에 대한 환불을 처리합니다.

    Args:
        amount: 환불 금액(원)
        reason: 환불 사유
    """
    refund_id = f"RFND-{random.randint(1000, 9999)}"
    return (
        f"💳 환불이 접수되었습니다. 환불 번호: {refund_id}, 금액: {amount:,}원, "
        f"사유: {reason}. 영업일 기준 3~5일 내에 처리됩니다."
    )


@function_tool
def request_manager_callback(name: str, phone: str, summary: str) -> str:
    """매니저가 손님에게 직접 연락하도록 콜백을 요청합니다.

    Args:
        name: 손님 이름
        phone: 연락처
        summary: 불만 요약
    """
    ticket = f"MGR-{random.randint(1000, 9999)}"
    return (
        f"📞 매니저 콜백이 예약되었습니다. 접수 번호: {ticket}. "
        f"담당 매니저가 24시간 이내에 {phone} 로 연락드립니다."
    )


@function_tool
def escalate_complaint(severity: str, summary: str) -> str:
    """심각한 불만을 상위 부서로 에스컬레이션합니다.

    Args:
        severity: 심각도 (low / medium / high / critical)
        summary: 불만 요약
    """
    case = f"ESC-{random.randint(10000, 99999)}"
    eta = {"critical": "1시간", "high": "4시간"}.get(severity.lower(), "24시간")
    return (
        f"🚨 불만이 에스컬레이션되었습니다. 케이스 번호: {case}, "
        f"심각도: {severity.upper()}, 예상 대응: {eta} 이내."
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
    - 음식·서비스에 대한 불만, 불만족, 환불 요청, 매니저 연결 요청 → Complaints Agent

    손님이 불만을 표현하면 먼저 짧게 공감하고 사과한 뒤 Complaints Agent 로 연결해.
    핸드오프 직전에 한국어로 짧게 어디로 연결하는지 안내하고 바로 해당 담당에게 넘겨.
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
    - 주문은 Order Agent, 예약은 Reservation Agent, 불만은 Complaints Agent 로 핸드오프해.
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
    - 메뉴/재료 질문은 Menu Agent, 예약은 Reservation Agent, 불만은 Complaints Agent 로 핸드오프해.
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
    - 메뉴/재료 질문은 Menu Agent, 주문은 Order Agent, 불만은 Complaints Agent 로 핸드오프해.
    항상 한국어로 친근하게 답해.
    """


def complaints_instructions(
    wrapper: RunContextWrapper[CustomerContext],
    agent: Agent,
):
    return f"""{RECOMMENDED_PROMPT_PREFIX}

    너는 '노마드 비스트로'의 고객 불만 처리 담당이야.
    손님({wrapper.context.name})의 불만을 세심하고 공감하는 태도로 처리해.

    처리 절차:
    1) 먼저 진심으로 공감하고 사과해. 손님의 불편한 경험을 분명히 인정해.
    2) 상황을 바로잡을 해결책을 제안해:
       - offer_discount: 다음 방문 시 할인 쿠폰 제공
       - process_refund: 환불 처리
       - request_manager_callback: 매니저가 직접 콜백
    3) 심각하거나 안전과 관련된 문제(식중독, 위생, 부상 등)는 escalate_complaint 로 에스컬레이션해.
    4) 어떤 해결책을 원하시는지 정중히 여쭤보고 진행해.

    절대 변명하거나 손님을 탓하지 마. 항상 한국어로 따뜻하고 정중하게 답해.
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
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

order_agent = Agent(
    name="Order Agent",
    handoff_description="음식 주문 접수 및 확인 담당",
    instructions=order_instructions,
    tools=[place_order],
    hooks=ToolLoggingHooks(),
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

reservation_agent = Agent(
    name="Reservation Agent",
    handoff_description="테이블 예약 처리 담당",
    instructions=reservation_instructions,
    tools=[check_availability, make_reservation],
    hooks=ToolLoggingHooks(),
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

complaints_agent = Agent(
    name="Complaints Agent",
    handoff_description="불만족한 고객의 불만 처리, 해결책 제시(환불·할인·매니저 콜백), 에스컬레이션 담당",
    instructions=complaints_instructions,
    tools=[offer_discount, process_refund, request_manager_callback, escalate_complaint],
    hooks=ToolLoggingHooks(),
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

triage_agent = Agent(
    name="Triage Agent",
    handoff_description="요청을 파악해 알맞은 전문 담당으로 연결",
    instructions=triage_instructions,
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

# 에이전트 간 상호 핸드오프 연결 (생성 후 설정해 순환 참조를 피한다).
# 어느 담당과 대화 중이어도 다른 전문가에게 바로 넘어갈 수 있다.
triage_agent.handoffs = [menu_agent, order_agent, reservation_agent, complaints_agent]
menu_agent.handoffs = [triage_agent, order_agent, reservation_agent, complaints_agent]
order_agent.handoffs = [triage_agent, menu_agent, reservation_agent, complaints_agent]
reservation_agent.handoffs = [triage_agent, menu_agent, order_agent, complaints_agent]
complaints_agent.handoffs = [triage_agent, menu_agent, order_agent, reservation_agent]


# 핸드오프 발생 시 UI 에 보여줄 한국어 안내 문구.
HANDOFF_NOTICE = {
    "Menu Agent": "🍽️ 메뉴 전문가에게 연결합니다...",
    "Order Agent": "📝 주문 담당에게 연결합니다...",
    "Reservation Agent": "📅 예약 담당에게 연결합니다...",
    "Complaints Agent": "🙇 불만 처리 담당에게 연결합니다...",
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

        try:

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

        except InputGuardrailTripwireTriggered as e:
            # 주제 이탈 / 부적절한 언어 → 정중히 거부
            inappropriate = False
            try:
                inappropriate = e.guardrail_result.output.output_info.is_inappropriate
            except Exception:
                pass

            if inappropriate:
                st.warning(
                    "정중한 대화를 부탁드려요. 🙏 저는 메뉴 확인, 예약, 주문, "
                    "불만 접수를 도와드릴 수 있어요."
                )
            else:
                st.warning(
                    "저는 레스토랑 관련 질문에 대해서만 도와드리고 있어요. "
                    "메뉴를 확인하거나, 예약하거나, 음식을 주문할 수 있어요."
                )

        except OutputGuardrailTripwireTriggered:
            # 비전문적 응답 / 내부 정보 노출 → 응답 차단
            st.session_state["text_placeholder"].empty()
            st.warning(
                "죄송해요, 적절한 답변을 준비하지 못했어요. "
                "다시 시도해 주시거나 다른 방식으로 질문해 주세요."
            )


message = st.chat_input("무엇을 도와드릴까요? (메뉴 · 주문 · 예약 · 불만)")

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
        "- 📅 **Reservation** — 테이블 예약\n"
        "- 🙇 **Complaints** — 불만 처리 · 해결책 제시"
    )
    st.divider()
    st.subheader("🛡️ Guardrails")
    st.markdown(
        "- **Input** — 주제 이탈 · 부적절한 언어 차단\n"
        "- **Output** — 비전문적 응답 · 내부 정보 노출 차단"
    )
    st.divider()
    if st.button("대화 초기화"):
        asyncio.run(session.clear_session())
        st.session_state["agent"] = triage_agent
        st.rerun()
