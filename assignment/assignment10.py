import base64

import dotenv

dotenv.load_dotenv()

from pydantic import BaseModel, Field
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from openai import OpenAI

from .prompt import (
    STORY_WRITER_DESCRIPTION,
    STORY_WRITER_INSTRUCTION,
    ILLUSTRATOR_DESCRIPTION,
    ILLUSTRATOR_INSTRUCTION,
)

MODEL = LiteLlm(model="openai/gpt-4o")


# ============================================================================
# 구조화된 스토리 데이터 (Story Writer 의 output_schema)
# ============================================================================


class StoryPage(BaseModel):
    page_number: int = Field(description="페이지 번호 (1~5)")
    text: str = Field(description="그 페이지의 동화 문장 (한국어)")
    visual: str = Field(description="삽화 설명 (영어, 이미지 생성용)")


class StoryBook(BaseModel):
    title: str = Field(description="동화책 제목")
    pages: list[StoryPage] = Field(description="정확히 5개의 페이지")


# ============================================================================
# Story Writer Agent — 테마 → 구조화된 5페이지 동화. 결과를 state["story"] 에 저장.
# ============================================================================

story_writer_agent = Agent(
    name="StoryWriterAgent",
    model=MODEL,
    description=STORY_WRITER_DESCRIPTION,
    instruction=STORY_WRITER_INSTRUCTION,
    output_schema=StoryBook,
    output_key="story",
)


# ============================================================================
# Illustrator Agent — state["story"] 를 읽어 페이지별 이미지를 Artifact 로 저장.
# ============================================================================


async def generate_image(
    tool_context: ToolContext, page_number: int, visual_description: str
):
    """한 페이지의 어린이 동화 삽화를 생성하고 Artifact 로 저장한다.

    Args:
        page_number: 페이지 번호 (1~5).
        visual_description: 삽화에 그릴 장면 설명.
    """
    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1",
        prompt=(
            "Children's storybook illustration, soft watercolor style, warm and "
            "friendly, bright gentle colors, simple shapes, no text in the image. "
            "Scene: " + visual_description
        ),
        size="1024x1024",
        n=1,
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)

    artifact = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    filename = f"page_{page_number}.png"
    version = await tool_context.save_artifact(filename=filename, artifact=artifact)

    return {
        "status": "success",
        "page": page_number,
        "artifact_filename": filename,
        "version": version,
    }


illustrator_agent = Agent(
    name="IllustratorAgent",
    model=MODEL,
    description=ILLUSTRATOR_DESCRIPTION,
    instruction=ILLUSTRATOR_INSTRUCTION,
    tools=[generate_image],
)


# ============================================================================
# Root Agent — 작가 → 삽화가 순서로 실행 (State 로 스토리 데이터 공유)
# ============================================================================

root_agent = SequentialAgent(
    name="StorybookAgent",
    description=(
        "테마로부터 5페이지 어린이 동화책을 만든다. 먼저 작가가 이야기를 쓰고, "
        "이어서 삽화가가 각 페이지의 이미지를 생성한다."
    ),
    sub_agents=[story_writer_agent, illustrator_agent],
)
