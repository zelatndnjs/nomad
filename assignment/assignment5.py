import dotenv

dotenv.load_dotenv()
import os
import asyncio
import streamlit as st
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool

st.set_page_config(page_title="Life Coach", page_icon="🌱")

client = OpenAI()

GOALS_FILE = "goals.txt"
VECTOR_STORE_ID_FILE = "vector_store_id.txt"
GOALS_MARKER_FILE = ".goals_uploaded"


def get_vector_store_id():
    """벡터 스토어 ID 를 로컬 파일에서 읽고, 없으면 새로 만들어 저장한다."""
    if os.path.exists(VECTOR_STORE_ID_FILE):
        with open(VECTOR_STORE_ID_FILE, "r") as f:
            vector_store_id = f.read().strip()
            if vector_store_id:
                return vector_store_id

    vector_store = client.vector_stores.create(name="life-coach-goals")
    with open(VECTOR_STORE_ID_FILE, "w") as f:
        f.write(vector_store.id)
    return vector_store.id


def ensure_goals_uploaded(vector_store_id):
    """목표 문서를 벡터 스토어에 한 번만 업로드하고 색인이 끝날 때까지 기다린다."""
    if os.path.exists(GOALS_MARKER_FILE):
        return
    if not os.path.exists(GOALS_FILE):
        return

    with open(GOALS_FILE, "rb") as f:
        uploaded_file = client.files.create(
            file=(GOALS_FILE, f.read()),
            purpose="user_data",
        )
    client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store_id,
        file_id=uploaded_file.id,
    )
    with open(GOALS_MARKER_FILE, "w") as f:
        f.write(uploaded_file.id)


if "vector_store_id" not in st.session_state:
    st.session_state["vector_store_id"] = get_vector_store_id()
    ensure_goals_uploaded(st.session_state["vector_store_id"])
VECTOR_STORE_ID = st.session_state["vector_store_id"]


if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions="""
        You are a warm and encouraging life coach.

        Your job is to motivate the user, help them build better habits and
        support their personal growth. Stay positive, supportive and practical,
        and always reply in Korean.

        You have access to the followign tools:
            - File Search Tool: 사용자의 개인 목표, 일기, 과거 기록에 대한 질문에는 먼저 이 도구로 업로드된 목표 문서를 검색하세요. 진행 상황을 조언하거나 평가할 때는 항상 목표 문서를 참고해 사용자가 실제로 세운 목표와 지난 기록을 확인하세요.
            - Web Search Tool: 질문이 학습 데이터에 없거나, 최신 정보·연구·실천 방법이 필요할 때 웹에서 검색하세요.

        조언할 때는 이렇게 하세요:
            1. 먼저 File Search Tool 로 사용자의 목표 문서를 검색해 목표와 과거 진행 상황을 확인합니다.
            2. 필요하면 Web Search Tool 로 검증된 최신 방법을 찾습니다.
            3. 사용자의 목표와 지난 기록, 그리고 검색한 정보를 결합해 그 사람에게 딱 맞는 개인화된 조언을 제공합니다.
        """,
        tools=[
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
            WebSearchTool(),
        ],
    )
agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "life-coach-memory.db",
    )
session = st.session_state["session"]


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", r"\$"))
        if "type" in message:
            if message["type"] == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 웹을 검색했어요...")
            elif message["type"] == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ 목표 문서를 검색했어요...")


def update_status(status_container, event):

    status_messages = {
        "response.web_search_call.completed": ("✅ 웹 검색 완료.", "complete"),
        "response.web_search_call.in_progress": (
            "🔍 웹 검색을 시작합니다...",
            "running",
        ),
        "response.web_search_call.searching": (
            "🔍 웹 검색 중...",
            "running",
        ),
        "response.file_search_call.completed": (
            "✅ 목표 문서 검색 완료.",
            "complete",
        ),
        "response.file_search_call.in_progress": (
            "🗂️ 목표 문서 검색을 시작합니다...",
            "running",
        ),
        "response.file_search_call.searching": (
            "🗂️ 목표 문서 검색 중...",
            "running",
        ),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


asyncio.run(paint_history())


async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        response = ""

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":

                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", r"\$"))


prompt = st.chat_input(
    "오늘 어떤 고민이 있나요?",
    accept_file=True,
    file_type=["txt"],
)

if prompt:

    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ 파일 업로드 중...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ 파일 첨부 중...")
                    client.vector_stores.files.create_and_poll(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ 파일 업로드 완료", state="complete")

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


with st.sidebar:
    st.markdown("**목표 문서**: `goals.txt`")
    st.caption("앱을 처음 실행하면 목표 문서가 자동으로 업로드됩니다. 새 .txt 파일을 채팅창에 첨부해 목표를 추가할 수도 있어요.")
    reset = st.button("기억 초기화")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
