"""Module for managing chat message history."""

from datetime import datetime
from typing import Any

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.chat_message_histories.sql import DefaultMessageConverter
from omegaconf.dictconfig import DictConfig
from sqlalchemy import Column, DateTime, Integer, Text

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

TABLE_NAME = "message_history"


def get_chat_message_history(config: DictConfig, chat_id: str) -> SQLChatMessageHistory:
    """Get the chat message history for a given chat id.

    Args:
        config (DictConfig): Configuration object.
        chat_id (str): Chat id.

    Returns:
        SQLChatMessageHistory: Chat message history.
    """
    return SQLChatMessageHistory(
        session_id=chat_id,
        connection_string=config.rag.database_url,
        table_name=TABLE_NAME,
        custom_message_converter=TimestampedMessageConverter(TABLE_NAME),
    )


class TimestampedMessageConverter(DefaultMessageConverter):
    """Message converter that adds a timestamp to the message."""

    def __init__(self, table_name: str) -> None:
        """Initialize the message converter."""
        self.model_class = create_message_model(table_name, declarative_base())


def create_message_model(table_name: str, dynamic_base: Any) -> Any:  # noqa: ANN401
    """Create a message model.

    Args:
        table_name (str): Name of the table.
        dynamic_base (Any): Dynamic base class.

    Returns:
        Any: Message model.
    """

    class Message(dynamic_base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        session_id = Column(Text)
        message = Column(Text)

    return Message
