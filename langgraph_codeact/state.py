from typing import Any, Optional

from langgraph.graph import MessagesState


class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: Optional[str]
    """The Python code script to be executed."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""
