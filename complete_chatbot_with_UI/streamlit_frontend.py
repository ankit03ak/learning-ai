import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# with st.chat_message("user"):
#         st.text("Hi")
# with st.chat_message("assistant"):
#         st.text("Hi! How can I help you today?")

CONFIG = {'configurable' : {'thread_id': 'thread-1'} }

# for history in st.session_state["messages"]:
if 'message_history' not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_response = response['messages'][-1].content
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.text(ai_response)