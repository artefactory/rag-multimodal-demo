from urllib.parse import urljoin

import streamlit as st
from requests.sessions import Session

from frontend import BACKEND_URL


def query(verb: str, url: str, **kwargs):
    session = st.session_state.get("authenticated_session")
    session = st.session_state["session"] if session is None else session

    response = getattr(session, verb)(url, **kwargs)

    if response.status_code == 401:
        st.session_state["authenticated_session"] = None
        st.session_state["email"] = None
        st.session_state["login_status_level"] = "error"
        st.session_state["login_status_message"] = (
            "Session expired. Please log in again."
        )
        st.rerun()

    return response


def backend_supports_sessions() -> bool:
    return query("get", BACKEND_URL + "session").status_code == 200


def backend_supports_auth() -> bool:
    return query("get", BACKEND_URL + "user").status_code == 200


def create_session() -> Session:
    session = BaseUrlSession(BACKEND_URL)
    return session


class BaseUrlSession(Session):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url

    def request(self, method, url, *args, **kwargs):
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        if url.startswith("/"):
            url = url[1:]
        url = urljoin(self.base_url, url)
        return super().request(method, url, *args, **kwargs)
