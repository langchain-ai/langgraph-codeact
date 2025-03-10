import inspect
from collections import ChainMap
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, MessageLikeRepresentation
from langchain_core.tools import Tool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.store.base import BaseStore


DEFAULT_PROMPT = """You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer."""


def create_codeact(
    tools: Sequence[Tool],
    model: BaseChatModel,
    eval: Callable[[str, dict[str, Callable]], str],
    *,
    prompt: str = DEFAULT_PROMPT,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    config_schema: Optional[type[Any]] = None,
):
    # create the prompt
    prompt = """
{prompt}

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in tools:
        prompt += f'''
def {tool.name}{str(inspect.signature(tool.func))}:
    """{tool.description}"""
    ...
'''

    prompt += """

Variables defined at the top level of previous code snippets can be referenced in your code."""

    @task
    def agent(
        messages: Sequence[MessageLikeRepresentation],
    ) -> tuple[AIMessage, Optional[str]]:
        """Calls model for next script or answer."""
        msg = model.invoke(messages)
        # extract code block
        if "```" in msg.content:
            # get content between fences
            code = msg.content.split("```")[1]
            # remove first line, which is the language or empty string
            code = "\n".join(code.splitlines()[1:])
            return msg, code
        else:
            # no code block, return None
            return msg, None

    @task
    def sandbox(script: str, context: dict[str, Callable]) -> str:
        """Executes the script in a sandboxed environment."""
        # execute the script
        return eval(script, context)

    @entrypoint(checkpointer=checkpointer, store=store, config_schema=config_schema)
    def codeact(
        task: str, *, previous: Optional[tuple[list[BaseMessage], dict[str, Any]]]
    ) -> str:
        # will accumulate messages
        msgs = [("system", prompt)]
        # will accumulate variables defined at script top-level
        locs = {}
        # contains locals + tools
        context = ChainMap(locs, {tool.name: tool.func for tool in tools})
        # add previous turn
        if previous is not None:
            prev_msgs, prev_locals = previous
            msgs.extend(prev_msgs)
            locs.update(prev_locals)
        # add task to messages
        msgs.append(("user", task))
        while True:
            # call agent
            msg, script = agent(msgs).result()
            # add message to history
            msgs.append(msg)
            if script is not None:
                output = sandbox(script, context).result()
                # add script output to messages
                msgs.append(("user", output))
            else:
                return msg.content

    return codeact
