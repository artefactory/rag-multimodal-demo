import streamlit as st
from langserve import RemoteRunnable
from PIL import Image

from frontend import ASSETS_PATH, BACKEND_URL_RAG
from frontend.lib.auth import authentication_page, create_session
from frontend.lib.backend_interface import (
    backend_supports_auth,
    backend_supports_sessions,
)
from frontend.lib.basic_chat import basic_chat
from frontend.lib.session_chat import session_chat
from frontend.lib.sidebar import sidebar
from frontend.lib.streamlit_helpers import initialize_state_variable


def browser_tab_title():
    st.set_page_config(
        page_title="RAG ChatBot",
        page_icon=Image.open(ASSETS_PATH / "logo_tab.jpeg"),
    )


def application_header():
    st.image(Image.open(ASSETS_PATH / "logo_title.jpeg"))
    st.caption(
        "Learn more about the RAG indus kit here: https://artefactory.github.io/skaff-rag-accelerator"
    )


if __name__ == "__main__":
    browser_tab_title()
    application_header()

    # The session is used to make requests to the backend. It helps with the handling of cookies, auth, and other session data
    initialize_state_variable("session", value=create_session())

    # The chain is our RAG that will be used to answer questions.
    # Langserve's RemoteRunnable allows us to work as if the RAG was local, but it's actually running on the backend
    initialize_state_variable("chain", value=RemoteRunnable(BACKEND_URL_RAG))

    # If the backend supports authentication but the user is not authenticated, show the authentication page
    if (
        backend_supports_auth()
        and st.session_state.get("authenticated_session", None) is None
    ):
        initialize_state_variable("login_status_message", value="")
        initialize_state_variable("login_status_level", value="info")
        authentication_page()
        st.stop()  # Stop the script to avoid running the rest of the code if the user is not authenticated

    # If the backend does not support authentication, just use the session as the authenticated session
    if (
        not backend_supports_auth()
        and st.session_state.get("authenticated_session", None) is None
    ):
        st.session_state["authenticated_session"] = st.session_state["session"]

    # Once we have an authenticated session, show the chat interface
    if st.session_state.get("authenticated_session") is not None:
        # If the backend supports sessions, enable session navigation
        if backend_supports_sessions():
            initialize_state_variable(
                "email", value="demo.user@email.com"
            )  # With authentication, this will take the user's email
            sidebar()
            session_chat()
        else:
            basic_chat()
