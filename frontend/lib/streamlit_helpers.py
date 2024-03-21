import streamlit as st


def initialize_state_variable(name: str, value):
    if name not in st.session_state:
        st.session_state[name] = value
