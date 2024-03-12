import time


def format_time_delta(delta: float) -> str:
    """Format a time delta in seconds to a string in the format HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.gmtime(delta))
