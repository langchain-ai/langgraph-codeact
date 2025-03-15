import inspect
from collections import ChainMap
from typing import Any, Callable, Optional, Sequence, Callable, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, MessageLikeRepresentation
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import Command
from typing import Literal


class CodeActState(MessagesState):
    """state for codeact agent."""

    script: str
    context: dict


def create_codeact(
    model: BaseChatModel,
    tools: Sequence[Union[BaseTool, Callable]],
    eval_fn: Callable[[str, dict[str, Callable]], str],
    *,
    prompt: Optional[str] = None,
):
    _tools = [t if isinstance(t, BaseTool) else create_tool(t) for t in tools]
    # create the prompt
    prompt = f"""
{prompt or ""}

You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in _tools:
        prompt += f'''
def {tool.name}{str(inspect.signature(tool.func))}:
    """{tool.description}"""
    ...
'''

    prompt += """

Variables defined at the top level of previous code snippets can be referenced in your code.

Reminder: use python code snippets to call tools"""

    def call_model(state: CodeActState) -> Command[Literal["__end__", "sandbox"]]:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        msg = model.invoke(messages)
        if "```" in msg.content:
            # get content between fences
            code = msg.content.split("```")[1]
            # remove first line, which is the language or empty string
            code = "\n".join(code.splitlines()[1:])
            return Command(goto="sandbox", update={"messages": [msg], "script": code})
        else:
            # no code block, return None
            return Command(goto="__end__", update={"messages": [msg], "script": ""})

    def sandbox(state: CodeActState):
        script = state["script"]
        context = state.get("context", {})
        # execute the script
        output = eval_fn(script, context)
        return {"messages": [{"role": "user", "content": output}]}

    agent = StateGraph(CodeActState)
    agent.add_node(call_model)
    agent.add_node(sandbox)
    agent.add_edge("__start__", "call_model")
    agent.add_edge("sandbox", "call_model")
    return agent
