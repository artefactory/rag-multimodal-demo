from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import streamlit as st

from frontend.lib.backend_interface import query


@dataclass
class Message:
    sender: str
    content: str
    session_id: str
    id: str = None
    timestamp: str = None

    def __post_init__(self):
        self.id = str(uuid4()) if self.id is None else self.id
        self.timestamp = (
            datetime.utcnow().isoformat() if self.timestamp is None else self.timestamp
        )


def session_chat():
    user_question = st.chat_input("Say something")

    with st.container(border=True):
        for message in st.session_state.get("messages", []):
            with st.chat_message(message.sender):
                st.write(message.content)

        if user_question:
            if len(st.session_state.get("messages", [])) == 0:
                session_id = new_session()
            else:
                session_id = st.session_state.get("session_id")

            with st.chat_message("user"):
                st.write(user_question)

            user_message = Message("user", user_question, session_id)
            st.session_state["messages"].append(user_message)

            chain = st.session_state.get("chain")
            response = chain.stream(
                {"question": user_question},
                {"configurable": {"session_id": session_id}},
            )

            with st.chat_message("assistant"):
                full_response = ""

                placeholder = st.empty()
                for chunk in response:
                    full_response += chunk.content
                    placeholder.write(full_response)

            bot_message = Message("assistant", full_response, session_id)
            st.session_state["messages"].append(bot_message)


def new_session():
    session_id = query("post", "/session/new").json()["session_id"]
    st.session_state["session_id"] = session_id
    st.session_state["messages"] = []
    return session_id


def get_session(session_id: str):
    session = query("get", f"/session/{session_id}").json()
    return session
