import inspect
from typing import Any, Callable, Optional, Sequence, Union, Literal, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.types import Command


class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: str
    """The Python code script to be executed."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""


def create_codeact(
    model: BaseChatModel,
    tools: Sequence[Union[BaseTool, Callable]],
    eval_fn: Callable[[str, dict[str, Callable]], tuple[str, dict]],
    *,
    prompt: Optional[str] = None,
) -> StateGraph:
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

    # Make tools available to the code sandbox
    tools_context = {tool.name: tool.func for tool in _tools}

    def call_model(state: CodeActState) -> Command:
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
            return Command(update={"messages": [msg], "script": ""})

    def sandbox(state: CodeActState):
        script = state["script"]
        existing_context = state.get("context", {})
        context = {**existing_context, **tools_context}
        # Execute the script in the sandbox
        output, new_vars = eval_fn(script, context)
        new_context = {**existing_context, **new_vars}
        return {
            "messages": [{"role": "user", "content": output}],
            "context": new_context,
        }

    agent = StateGraph(CodeActState)
    agent.add_node(call_model, destinations=(END, "sandbox"))
    agent.add_node(sandbox)
    agent.add_edge(START, "call_model")
    agent.add_edge("sandbox", "call_model")
    return agent
