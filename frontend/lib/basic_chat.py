import streamlit as st


def basic_chat():
    user_question = st.chat_input("Say something")

    with st.container(border=True):
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

            chain = st.session_state.get("chain")
            response = chain.stream(user_question)

            with st.chat_message("assistant"):
                st.write(response)
