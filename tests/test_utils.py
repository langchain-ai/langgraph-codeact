from langgraph_codeact.utils import extract_and_combine_codeblocks


def test_empty_text():
    """Test when the input text has no codeblocks."""
    text = "This is a text without any code blocks."
    result = extract_and_combine_codeblocks(text)
    assert result == ""


def test_single_codeblock_no_language():
    """Test extracting a single codeblock without language identifier."""
    text = """Here is a code block:
```
print("Hello, world!")
x = 10
```
End of the code."""

    expected = """\
print("Hello, world!")
x = 10\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected


def test_single_codeblock_with_language():
    """Test extracting a single codeblock with language identifier."""
    text = """Here is a code block:
```python
print("Hello, world!")
x = 10
```
End of the code."""

    expected = """\
print("Hello, world!")
x = 10\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected


def test_multiple_codeblocks():
    """Test extracting and combining multiple codeblocks."""
    text = """Here's the first code block:
```python
def hello():
    print("Hello!")
```

And here's the second one:
```python
result = 42
print(f"The answer is {result}")
```"""

    expected = """\
def hello():
    print("Hello!")

result = 42
print(f"The answer is {result}")\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected


def test_multiple_codeblocks_mixed():
    """Test codeblocks with a mix of language identifiers / no identifiers."""
    text = """Different language identifiers:
```python
x = 10
```

```python
y = 20
```

```
z = 30
```"""

    expected = """\
x = 10

y = 20

z = 30\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected


def test_empty_codeblock():
    """Test an empty codeblock."""
    text = "Empty block: `````` should be ignored."
    result = extract_and_combine_codeblocks(text)
    assert result == ""


def test_language_with_spaces():
    """Test a codeblock with a language identifier containing spaces."""
    text = """Here is code with a more unusual language tag:
```python code
x = 10
y = 20
```"""

    # The first line shouldn't be removed since it contains spaces
    expected = """\
python code
x = 10
y = 20\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected


def test_with_nested_backticks():
    """Test with nested backticks inside the code block."""
    text = """Code with nested backticks:
```
def example():
    code = "```nested```"
    return code
```"""

    expected = """\
def example():
    code = "```nested```"
    return code\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected


def test_realistic_example():
    """Test with a realistic example similar to the one provided in the user query."""
    text = """First, I'll find where the baseball lands when hit by the batter. Then, I'll calculate where the ball lands after being thrown by the outfielder.

```python
# Constants
g = 9.81  # acceleration due to gravity
v0_batter = 45.847  # initial velocity
angle_batter_deg = 23.474  # angle in degrees

print(f"The ball lands {distance:.2f} meters away")
```

Now, let's calculate the second trajectory:

```
# Outfielder's throw
v0_outfielder = 24.12  # initial velocity
distance_2 = v0_outfielder * 2  # simplified calculation
print(f"Final position: {distance_2:.2f} meters")
```"""

    expected = """\
# Constants
g = 9.81  # acceleration due to gravity
v0_batter = 45.847  # initial velocity
angle_batter_deg = 23.474  # angle in degrees

print(f"The ball lands {distance:.2f} meters away")

# Outfielder's throw
v0_outfielder = 24.12  # initial velocity
distance_2 = v0_outfielder * 2  # simplified calculation
print(f"Final position: {distance_2:.2f} meters")\
"""
    result = extract_and_combine_codeblocks(text)
    assert result == expected
