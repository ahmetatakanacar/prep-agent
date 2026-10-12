"""The fixed request behind `prep-agent llm-check`."""

from pydantic import BaseModel, Field

PING_SYSTEM = "You are a connectivity check. Answer with the single word pong."
PING_USER = "ping"


class PingReply(BaseModel):
    reply: str = Field(description="The single word pong.")
