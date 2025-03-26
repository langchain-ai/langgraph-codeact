# langgraph-codeact

This library implements the [CodeAct architecture](https://arxiv.org/abs/2402.01030) in LangGraph. This is the architecture is used by [Manus.im](https://manus.im/). It implements an alternative to JSON function-calling, which enables solving more complex tasks in less steps. This is achieved by making use of the full power of a Turing complete programming language (such as Python used here) to combine and transform the outputs of multiple tools.

## Features

- Message history is saved between turns, to support follow-up questions
- Python variables are saved between turns, which enables more advanced follow-up questions
- Use .invoke() to get just the final result, or .stream() to get token-by-token output, see example below
- You can use any custom tools you wrote, any LangChain tools, or any MCP tools
- You can use this with any model supported by LangChain (but we've only tested with Claude 3.7 so far)
- You can bring your own code sandbox, with a simple functional API
- The system message is customizable

## Installation

```bash
pip install langgraph-codeact
```

To run the example install also

```bash
pip install langchain langchain-anthropic
```

## Example

A full version of this in one file can be found [here](examples/math_example.py)

### 1. Define your tools

You can use any tools you want, including custom tools, LangChain tools, or MCP tools. In this example, we define a few simple math functions.

```py
import math

from langchain_core.tools import tool

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

def sin(a: float) -> float:
    """Take the sine of a number."""
    return math.sin(a)

def cos(a: float) -> float:
    """Take the cosine of a number."""
    return math.cos(a)

def radians(a: float) -> float:
    """Convert degrees to radians."""
    return math.radians(a)

def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b

def sqrt(a: float) -> float:
    """Take the square root of a number."""
    return math.sqrt(a)

def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    return math.ceil(a)

tools = [
    add,
    multiply,
    divide,
    subtract,
    sin,
    cos,
    radians,
    exponentiation,
    sqrt,
    ceil,
]
```

### 2. Bring-your-own code sandbox

You can use any code sandbox you want, pass it in as a function which accepts two arguments

- the string of code to run
- the dictionary of locals to run it in (includes the tools, and any variables you set in the previous turns)

> [!Warning]
> Use a sandboxed environment in production! The `eval` function below is just for demonstration purposes, not safe!

```py
import builtins
import contextlib
import io
from typing import Any


def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars
```

### 3. Create the CodeAct graph

You can also customize the prompt, through the prompt= argument.

```py
from langchain.chat_models import init_chat_model
from langgraph_codeact import create_codeact
from langgraph.checkpoint.memory import MemorySaver

model = init_chat_model("claude-3-7-sonnet-latest", model_provider="anthropic")

code_act = create_codeact(model, tools, eval)
agent = code_act.compile(checkpointer=MemorySaver())
```

### 4. Run it!

You can use the `.invoke()` method to get the final result, or the `.stream()` method to get token-by-token output.

```py

messages = [{
    "role": "user",
    "content": "A batter hits a baseball at 45.847 m/s at an angle of 23.474° above the horizontal. The outfielder, who starts facing the batter, picks up the baseball as it lands, then throws it back towards the batter at 24.12 m/s at an angle of 39.12 degrees. How far is the baseball from where the batter originally hit it? Assume zero air resistance."
}]
for typ, chunk in agent.stream(
    {"messages": messages},
    stream_mode=["values", "messages"],
    config={"configurable": {"thread_id": 1}},
):
    if typ == "messages":
        print(chunk[0].content, end="")
    elif typ == "values":
        print("\n\n---answer---\n\n", chunk)
```
