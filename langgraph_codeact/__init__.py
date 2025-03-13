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


def create_codeact(
    tools: Sequence[Union[BaseTool, Callable]],
    model: BaseChatModel,
    eval: Callable[[str, dict[str, Callable]], str],
    *,
    prompt: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    config_schema: Optional[type[Any]] = None,
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
        state: dict
    ) -> str:
        # will accumulate variables defined at script top-level
        locs = state.get("locals", {})
        # contains locals + tools
        context = ChainMap(locs, {tool.name: tool.func for tool in _tools})
        
        # Get messages from state
        msgs = [{"role": "system", "content": prompt}] + state.get("messages", [])
        
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
