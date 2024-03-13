"""Common utility functions."""

import time


def format_time_delta(delta: float) -> str:
    """Format a time delta in seconds to a string in the format HH:MM:SS.

    Args:
        delta (float): Time delta in seconds.

    Returns:
        str: Time delta formatted.
    """
    return time.strftime("%H:%M:%S", time.gmtime(delta))
