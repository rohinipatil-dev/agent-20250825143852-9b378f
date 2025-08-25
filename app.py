import os
from typing import List, Dict

import streamlit as st
from openai import OpenAI


def get_system_messages(specialize_for_python: bool = True) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    if specialize_for_python:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "You are specialized in answering Python programming questions. "
                    "Provide clear, concise explanations and idiomatic Python 3 code. "
                    "Use fenced code blocks for examples, explain reasoning briefly, "
                    "and mention edge cases and best practices when relevant. "
                    "If a question is unrelated to Python, ask a gentle clarifying question to steer it back to Python."
                ),
            }
        )
    return msgs


def build_chat_messages(
    history: List[Dict[str, str]],
    system_messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Compose the full message list to send to the OpenAI API.
    """
    messages: List[Dict[str, str]] = []
    messages.extend(system_messages)
    for item in history:
        if item["role"] in ("user", "assistant"):
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


def generate_response(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """
    Call OpenAI Chat Completions API and return the assistant's message content.
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,  # "gpt-3.5-turbo" or "gpt-4"
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm your Python programming assistant. "
                    "Ask me anything about Python, from basics to advanced topics like async, typing, "
                    "packaging, performance, testing, data science, and more."
                ),
            }
        ]


def sidebar_controls():
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Choose the model for responses.",
    )
    temperature = st.sidebar.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values make answers more focused and deterministic.",
    )
    specialize = st.sidebar.checkbox(
        "Specialize for Python Q&A", value=True, help="Keep the assistant focused on Python."
    )
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []
        init_session_state()
        st.experimental_rerun()

    if st.sidebar.checkbox("Show system prompt", value=False):
        sys_msgs = get_system_messages(specialize_for_python=specialize)
        st.sidebar.code("\n\n".join(m['content'] for m in sys_msgs), language="markdown")

    return model, temperature, specialize


def main():
    st.set_page_config(page_title="Python Programming Chatbot", page_icon="🐍", layout="wide")
    st.title("🐍 Python Programming Chatbot")

    init_session_state()
    model, temperature, specialize = sidebar_controls()

    api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    if not api_key_present:
        st.warning(
            "OpenAI API key not found. Set the OPENAI_API_KEY environment variable to enable responses."
        )

    # Render chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a Python question...")
    if user_input:
        # Append user message to history and render
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare and send request if API key present
        assistant_reply = "I'm unable to respond because the OpenAI API key is missing."
        if api_key_present:
            try:
                system_messages = get_system_messages(specialize_for_python=specialize)
                full_messages = build_chat_messages(st.session_state.history, system_messages)
                assistant_reply = generate_response(
                    messages=full_messages,
                    model=model,
                    temperature=temperature,
                )
            except Exception as e:
                assistant_reply = f"An error occurred while generating a response: {e}"

        # Append and render assistant reply
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


if __name__ == "__main__":
    main()