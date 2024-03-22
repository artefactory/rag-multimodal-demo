from datetime import datetime

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.chat_message_histories.sql import DefaultMessageConverter
from sqlalchemy import Column, DateTime, Integer, Text
from omegaconf.dictconfig import DictConfig


try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

TABLE_NAME = "message_history"


def get_chat_message_history(config: DictConfig, chat_id):
    return SQLChatMessageHistory(
        session_id=chat_id,
        connection_string=config.rag.database_url,
        table_name=TABLE_NAME,
        custom_message_converter=TimestampedMessageConverter(TABLE_NAME),
    )


class TimestampedMessageConverter(DefaultMessageConverter):
    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())


def create_message_model(table_name, dynamic_base):
    class Message(dynamic_base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        session_id = Column(Text)
        message = Column(Text)

    return Message
