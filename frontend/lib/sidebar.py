from datetime import datetime

import humanize
import streamlit as st

from frontend.lib.backend_interface import query
from frontend.lib.session_chat import Message


def sidebar():
    with st.sidebar:
        st.sidebar.title("RAG Industrialization Kit", anchor="top")
        st.sidebar.markdown(
            f"<p style='color:grey;'>Logged in as {st.session_state['email']}</p>",
            unsafe_allow_html=True,
        )

        if st.sidebar.button(
            "New Chat", use_container_width=True, key="new_chat_button"
        ):
            st.session_state["messages"] = []

        with st.empty():
            chat_list = list_sessions()
            chats_by_time_ago = {}
            for chat in chat_list:
                chat_id, timestamp = chat["id"], chat["timestamp"]
                time_ago = humanize.naturaltime(
                    datetime.utcnow() - datetime.fromisoformat(timestamp)
                )
                if time_ago not in chats_by_time_ago:
                    chats_by_time_ago[time_ago] = []
                chats_by_time_ago[time_ago].append(chat)

            for time_ago, chats in chats_by_time_ago.items():
                st.sidebar.markdown(time_ago)
                for chat in chats:
                    chat_id = chat["id"]
                    chat_first_message = chat["first_message"]
                    label = (
                        truncate_label(chat_first_message, 100)
                        if chat_first_message
                        else "*No content*"
                    )
                    if st.sidebar.button(
                        label=label,
                        key=chat_id,
                        use_container_width=True,
                    ):
                        st.session_state["chat_id"] = chat_id
                        messages = [
                            Message(**message)
                            for message in get_session(chat_id)["messages"]
                        ]
                        st.session_state["messages"] = messages


def list_sessions():
    return query("get", "/session/list").json()


def get_session(session_id: str):
    session = query("get", f"/session/{session_id}").json()
    return session


def truncate_label(string: str, max_len: int) -> str:
    """Truncate a string to a maximum length, appending ellipsis if necessary.

    Args:
        string (str): String to be truncated.
        max_len (int): Maximum allowed length of the string after truncation.

    Returns:
        str: Truncated string.
    """
    if string and len(string) > max_len:
        string = string[: max_len - 3] + "..."
    return string
