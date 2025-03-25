import inspect
from typing import Any, Callable, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool, tool as create_tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.types import Command


class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: Optional[str]
    """The Python code script to be executed."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""


def create_default_prompt(
    tools: list[StructuredTool], base_prompt: Optional[str] = None
):
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += f"""You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in tools:
        prompt += f'''
def {tool.name}{str(inspect.signature(tool.func))}:
    """{tool.description}"""
    ...
'''

    prompt += """

Variables defined at the top level of previous code snippets can be referenced in your code.

Reminder: use Python code snippets to call tools"""
    return prompt


def create_codeact(
    model: BaseChatModel,
    tools: Sequence[Union[StructuredTool, Callable]],
    eval_fn: Callable[[str, dict[str, Callable]], tuple[str, dict]],
    *,
    prompt: Optional[str] = None,
) -> StateGraph:
    """Create a CodeAct agent.

    Args:
        model: The language model to use for generating code
        tools: List of tools available to the agent. Can be passed as python functions or StructuredTool instances.
        eval_fn: Function that executes code in a sandbox. Takes code string and locals dict,
            returns a tuple of (stdout output, new variables dict)
        prompt: Optional custom system prompt. If None, uses default prompt.
            To customize default prompt you can use `create_default_prompt` helper:
            `create_default_prompt(tools, "You are a helpful assistant.")`

    Returns:
        A StateGraph implementing the CodeAct architecture
    """
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]

    if prompt is None:
        prompt = create_default_prompt(tools)

    # Make tools available to the code sandbox
    tools_context = {tool.name: tool.func for tool in tools}

    def call_model(state: CodeActState) -> Command:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = model.invoke(messages)
        if "```" in response.content:
            # get content between fences
            code = response.content.split("```")[1]
            # remove first line, which is the language or empty string
            code = "\n".join(code.splitlines()[1:])
            return Command(
                goto="sandbox", update={"messages": [response], "script": code}
            )
        else:
            # no code block, end the loop and respond to the user
            return Command(update={"messages": [response], "script": None})

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
