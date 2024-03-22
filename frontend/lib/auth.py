from time import sleep

import extra_streamlit_components as stx
import requests
import streamlit as st
from langserve import RemoteRunnable

from frontend import ADMIN_MODE, BACKEND_URL_RAG
from frontend.lib.backend_interface import create_session


def authentication_page():
    auth_form_tabs = [stx.TabBarItemData(id="Login", title="Login", description="")]
    if ADMIN_MODE:
        auth_form_tabs += [
            stx.TabBarItemData(id="Signup", title="Signup", description="")
        ]

    tab = stx.tab_bar(data=auth_form_tabs, default="Login")
    if tab == "Login":
        login_form()
    elif tab == "Signup":
        signup_form()


def login_form():
    with st.form("Login"):
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password")
        if st.session_state["login_status_message"]:
            # Dynamically decides whether to call st.error, st.success, or any other Streamlit method
            getattr(st, st.session_state["login_status_level"])(
                st.session_state["login_status_message"]
            )
        submit = st.form_submit_button("Log in")

        if submit:
            session = None
            token = get_token(username, password)
            if token:
                session = create_session()
                session = authenticate_session(session, token)
            else:
                st.session_state["login_status_level"] = "error"
                st.session_state["login_status_message"] = (
                    "Username/password combination not found"
                )
            st.session_state["authenticated_session"] = session
            st.session_state["email"] = username
            st.rerun()


def signup_form():
    with st.form("Signup"):
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password")
        if st.session_state["login_status_message"]:
            getattr(st, st.session_state["login_status_level"])(
                st.session_state["login_status_message"]
            )
        submit = st.form_submit_button("Sign up")

        if submit:
            auth_session = None
            success = sign_up(username, password)
            if success:
                token = get_token(username, password)
                session = create_session()
                auth_session = authenticate_session(session, token)
                st.session_state["login_status_level"] = "success"
                st.session_state["login_status_message"] = "Success! Account created."
                if st.session_state["login_status_message"]:
                    getattr(st, st.session_state["login_status_level"])(
                        st.session_state["login_status_message"]
                    )
                sleep(1.5)
            else:
                st.session_state["login_status_level"] = "error"
                st.session_state["login_status_message"] = "Failed signing up"
            st.session_state["authenticated_session"] = auth_session
            st.session_state["email"] = username
            st.rerun()


def get_token(username: str, password: str) -> str | None:
    session = create_session()
    response = session.post(
        "/user/login", data={"username": username, "password": password}
    )
    if response.status_code == 200 and "access_token" in response.json():
        return response.json()["access_token"]
    else:
        return None


def sign_up(username: str, password: str) -> bool:
    session = create_session()
    response = session.post(
        "/user/signup", json={"email": username, "password": password}
    )
    if response.status_code == 200 and "email" in response.json():
        return True
    else:
        return False


def authenticate_session(session, bearer_token: str) -> requests.Session:
    session.headers.update({"Authorization": f"Bearer {bearer_token}"})
    st.session_state["chain"] = RemoteRunnable(
        BACKEND_URL_RAG, headers={"Authorization": f"Bearer {bearer_token}"}
    )
    return session
