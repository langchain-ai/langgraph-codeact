import inspect
from typing import Optional

from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as create_tool


def create_default_prompt(
    tools: list[StructuredTool], base_prompt: Optional[str] = None
) -> str:
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution.
Any output you want to extract from the code should be printed to the console.
Code should be output in a fenced code block.
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
