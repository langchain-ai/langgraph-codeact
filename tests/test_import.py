def test_import() -> None:
    """Test that the code can be imported"""
    from langgraph_codeact import (  # noqa: F401
        create_codeact,
        create_default_prompt,
    )
