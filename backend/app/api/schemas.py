"""Request and response models for the REST API."""

from pydantic import BaseModel, Field

from app.agents.base import Citation


class AgentSummary(BaseModel):
    """One registered agent, as `GET /v1/agents` reports it.

    `skills` is part of the contract because an agent is now defined by the
    tools it can call: showing them is how the UI can say what the assistant is
    able to do without that list being maintained in two places.
    """

    name: str
    description: str
    skills: list[str] = []


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    agent: str
    citations: list[Citation] = []
    session_id: str
    trace_id: str


class SessionTurn(BaseModel):
    question: str
    answer: str
    agent: str
    at: str


class SessionResponse(BaseModel):
    session_id: str
    turns: list[SessionTurn] = []
    # History is bounded, so a client cannot otherwise tell a short
    # conversation from one whose oldest turns have been dropped.
    truncated: bool = False


class UploadResponse(BaseModel):
    filename: str
    doc_type: str
    chunks_indexed: int
