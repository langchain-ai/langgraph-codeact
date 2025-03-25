import base64
import builtins
import contextlib
import io

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph_codeact import create_codeact, create_default_prompt


def eval(code: str, _locals: dict) -> tuple:
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


def caesar_shift_decode(text: str, shift: int) -> str:
    """Decode text that was encoded using Caesar shift.

    Args:
        text: The encoded text to decode
        shift: The number of positions to shift back (positive number)

    Returns:
        The decoded text
    """
    result = ""
    for char in text:
        if char.isalpha():
            # Determine the case and base ASCII value
            ascii_base = ord("A") if char.isupper() else ord("a")
            # Shift the character back and wrap around if needed
            shifted = (ord(char) - ascii_base - shift) % 26
            result += chr(ascii_base + shifted)
        else:
            result += char
    return result


def base64_decode(text: str) -> str:
    """Decode text that was encoded using base64.

    Args:
        text: The base64 encoded text to decode

    Returns:
        The decoded text as a string

    Raises:
        Exception: If the input is not valid base64
    """
    try:
        # Add padding if needed
        padding = 4 - (len(text) % 4)
        if padding != 4:
            text += "=" * padding

        # Decode the base64 string
        decoded_bytes = base64.b64decode(text)
        return decoded_bytes.decode("utf-8")
    except Exception as e:
        raise Exception(f"Invalid base64 input: {str(e)}")


def caesar_shift_encode(text: str, shift: int) -> str:
    """Encode text using Caesar shift.

    Args:
        text: The text to encode
        shift: The number of positions to shift forward (positive number)

    Returns:
        The encoded text
    """
    result = ""
    for char in text:
        if char.isalpha():
            # Determine the case and base ASCII value
            ascii_base = ord("A") if char.isupper() else ord("a")
            # Shift the character forward and wrap around if needed
            shifted = (ord(char) - ascii_base + shift) % 26
            result += chr(ascii_base + shifted)
        else:
            result += char
    return result


def base64_encode(text: str) -> str:
    """Encode text using base64.

    Args:
        text: The text to encode

    Returns:
        The base64 encoded text as a string
    """
    # Convert text to bytes and encode
    text_bytes = text.encode("utf-8")
    encoded_bytes = base64.b64encode(text_bytes)
    return encoded_bytes.decode("utf-8")


# List of available tools
tools = [
    caesar_shift_decode,
    base64_decode,
    caesar_shift_encode,
    base64_encode,
]

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
code_act = create_codeact(
    model,
    tools,
    eval,
    prompt=create_default_prompt(
        tools,
        "Once you have the final answer, respond to the user with plain text, do not respond with a code snippet.",
    ),
)
agent = code_act.compile(checkpointer=MemorySaver())

if __name__ == "__main__":

    def stream_from_agent(messages: list[dict], config: RunnableConfig):
        for typ, chunk in agent.stream(
            {"messages": messages},
            stream_mode=["values", "messages"],
            config=config,
        ):
            if typ == "messages":
                print(chunk[0].content, end="")
            elif typ == "values":
                print("\n\n---answer---\n\n", chunk)

    # first turn
    config = {"configurable": {"thread_id": 1}}
    stream_from_agent(
        [
            {
                "role": "user",
                "content": "Decipher this text: 'VGhybCB6dnRsYW9wdW4gZHZ1a2x5bWJz'",
            }
        ],
        config,
    )

    # second turn
    stream_from_agent(
        [
            {
                "role": "user",
                "content": "Using the same cipher as the original text, encode this text: 'The work is mysterious and important'",
            }
        ],
        config,
    )
