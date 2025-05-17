import inspect
import json
import re
from typing import Any, Awaitable, Callable, Optional, Sequence, Type, TypeVar, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as create_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from langgraph_codeact.utils import extract_and_combine_codeblocks

EvalFunction = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
EvalCoroutine = Callable[[str, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]


class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: Optional[str]
    """The Python code script to be executed."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""


StateSchema = TypeVar("StateSchema", bound=CodeActState)
StateSchemaType = Type[StateSchema]


def make_safe_function_name(name: str) -> str:
    """Convert a tool name to a valid Python function name."""
    # Replace non-alphanumeric characters with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure the name doesn't start with a digit
    if safe_name and safe_name[0].isdigit():
        safe_name = f"tool_{safe_name}"
    # Handle empty name edge case
    if not safe_name:
        safe_name = "unnamed_tool"
    return safe_name


def create_default_prompt(tools: list[StructuredTool], base_prompt: Optional[str] = None):
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in tools:
        # Use coroutine if it exists, otherwise use func
        tool_callable = tool.coroutine if hasattr(tool, "coroutine") and tool.coroutine is not None else tool.func
        # Create a safe function name
        safe_name = make_safe_function_name(tool.name)
        # Determine if it's an async function
        is_async = inspect.iscoroutinefunction(tool_callable)
        # Add appropriate function definition
        prompt += f'''
{"async " if is_async else ""}def {safe_name}{str(inspect.signature(tool_callable))}:
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
    eval_fn: Union[EvalFunction, EvalCoroutine],
    *,
    prompt: Optional[str] = None,
    reflection_prompt: Optional[str] = None,
    reflection_model: Optional[BaseChatModel] = None,
    max_reflections: int = 3,
    state_schema: StateSchemaType = CodeActState,
) -> StateGraph:
    """Create a CodeAct agent.

    Args:
        model: The language model to use for generating code
        tools: List of tools available to the agent. Can be passed as python functions or StructuredTool instances.
        eval_fn: Function or coroutine that executes code in a sandbox. Takes code string and locals dict,
            returns a tuple of (stdout output, new variables dict)
        prompt: Optional custom system prompt. If None, uses default prompt.
            To customize default prompt you can use `create_default_prompt` helper:
            `create_default_prompt(tools, "You are a helpful assistant.")`
        reflection_prompt: Optional prompt for reflection. If provided, will be used to evaluate responses.
            If the reflection output contains "NONE", the response is considered valid, otherwise the
            reflection output is passed back to the model for regeneration.
        reflection_model: Optional model to use for reflection. If None, uses the same model as for generation.
        max_reflections: Maximum number of reflection iterations (default: 3).
        state_schema: The state schema to use for the agent.

    Returns:
        A StateGraph implementing the CodeAct architecture
    """
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]

    if prompt is None:
        prompt = create_default_prompt(tools)
        
    # If no reflection model is provided, use the main model
    if reflection_model is None:
        reflection_model = model

    # Make tools available to the code sandbox - use safe names for keys
    tools_context = {}
    for tool in tools:
        safe_name = make_safe_function_name(tool.name)
        # Use coroutine if it exists, otherwise use func (same as in create_default_prompt)
        tool_callable = tool.coroutine if hasattr(tool, "coroutine") and tool.coroutine is not None else tool.func
        # Only use the safe name for consistency with the prompt
        tools_context[safe_name] = tool_callable

    def call_model(state: StateSchema) -> Command:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        
        # Run the model and potentially loop for reflection
        response = model.invoke(messages)
        
        # Extract and combine all code blocks
        code = extract_and_combine_codeblocks(response.content)
        
        # Loop for reflection if needed and if code is present
        if reflection_prompt and code:
            reflection_count = 0
            while reflection_count < max_reflections:
                # Format conversation history with XML-style tags
                conversation_history = "\n".join([
                    f'<message role="{("user" if m.type == "human" else "assistant")}">\n{m.content}\n</message>'
                    for m in state["messages"]
                ])
                
                # Add the current response
                conversation_history += f'\n<message role="assistant">\n{response.content}\n</message>'
                
                # Create the reflection prompt with the tagged conversation history
                formatted_prompt = """
Review the assistant's latest code for as per the quality rules:

<conversation_history>
{}
</conversation_history>

If you find ANY of these issues, describe the problem briefly and clearly.
If NO issues are found, respond with EXACTLY: "NONE"
""".format(conversation_history)
                
                # Create messages for reflection with correct ordering
                reflection_messages = [
                    {"role": "system", "content": reflection_prompt},
                    # Include the formatted reflection prompt as the final user message
                    {"role": "user", "content": formatted_prompt}
                ]
                reflection_result = reflection_model.invoke(reflection_messages)
                
                # Check if reflection passed
                if "NONE" in reflection_result.content:
                    # Reflection passed, exit loop
                    break
                
                # Reflection didn't pass, regenerate response
                reflection_messages = [
                    {"role": "system", "content": prompt},
                    *state["messages"],
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": f"""
I need you to completely regenerate your previous response based on this feedback:

'''
{reflection_result.content}
'''

DO NOT reference the feedback directly. Instead, provide a completely new response that addresses the issues.
"""}
                ]
                response = model.invoke(reflection_messages)
                
                # Extract code from the new response
                code = extract_and_combine_codeblocks(response.content)
                
                # If no code in the new response, exit the reflection loop
                if not code:
                    break
                
                # Increment reflection count
                reflection_count += 1
        
        # Return appropriate command with only the latest response
        if code:
            return Command(goto="sandbox", update={"messages": [response], "script": code})
        else:
            # no code block, end the loop and respond to the user
            return Command(update={"messages": [response], "script": None})

    # If eval_fn is a async, we define async node function.
    if inspect.iscoroutinefunction(eval_fn):

        async def sandbox(state: StateSchema):
            existing_context = state.get("context", {})
            context = {**existing_context, **tools_context}
            # Execute the script in the sandbox
            output, new_vars = await eval_fn(state["script"], context)
            new_context = {**existing_context, **new_vars}
            return {
                "messages": [{"role": "user", "content": output}],
                "context": new_context,
            }
    else:

        def sandbox(state: StateSchema):
            existing_context = state.get("context", {})
            context = {**existing_context, **tools_context}
            # Execute the script in the sandbox
            output, new_vars = eval_fn(state["script"], context)
            new_context = {**existing_context, **new_vars}
            return {
                "messages": [{"role": "user", "content": output}],
                "context": new_context,
            }

    agent = StateGraph(state_schema)
    agent.add_node(call_model, destinations=(END, "sandbox"))
    agent.add_node(sandbox)
    agent.add_edge(START, "call_model")
    agent.add_edge("sandbox", "call_model")
    return agent
