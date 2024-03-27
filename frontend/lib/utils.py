"""Utility functions for the frontend."""

import re


def protect_dollar(string: str) -> str:
    r"""Escape unescaped dollar signs in a string.

    This function takes a string and returns a new string where all dollar signs ($)
    that are not preceded by a backslash (\) are escaped with an additional backslash.
    This is useful for preparing strings that contain dollar signs for environments
    where the dollar sign may be interpreted as a special character, such as in
    Markdown.

    Args:
        string (str): The input string containing dollar signs to be escaped.

    Returns:
        str: A new string with unescaped dollar signs preceded by a backslash.
    """
    return re.sub(r"(?<!\\)\$", r"\$", string)


def format_string(string: str) -> str:
    """Format a string for safe Markdown usage.

    Args:
        string (str): The input string to be formatted.

    Returns:
        str: The formatted string.
    """
    string = protect_dollar(string)
    return string
