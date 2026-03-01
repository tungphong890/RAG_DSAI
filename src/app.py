"""
Streamlit frontend with a ChatGPT-style chat layout.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_BACKEND_URL = os.environ.get("RAG_BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_REQUEST_TIMEOUT_S = int(os.environ.get("RAG_REQUEST_TIMEOUT_S", "120"))
DEFAULT_WELCOME_MESSAGE = "How can I help you today? Ask a question about data science, machine learning, or AI."


def _init_state() -> None:
    st.session_state.setdefault(
        "messages", [{"role": "assistant", "content": DEFAULT_WELCOME_MESSAGE}]
    )
    st.session_state.setdefault("backend_url", DEFAULT_BACKEND_URL)
    st.session_state.setdefault("mode", "hybrid")
    st.session_state.setdefault("top_k", 3)
    st.session_state.setdefault("reasoning", False)
    st.session_state.setdefault("show_metadata", True)
    st.session_state.setdefault("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)


def _reset_chat() -> None:
    st.session_state["messages"] = [
        {"role": "assistant", "content": DEFAULT_WELCOME_MESSAGE}
    ]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #343541;
            --bg-sidebar: #202123;
            --bg-assistant: #444654;
            --text-main: #ececf1;
            --text-muted: #a9a9b3;
            --border-color: #565869;
            --accent: #10a37f;
        }

        html, body, [class*="css"] {
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .stApp {
            background: var(--bg-main);
            color: var(--text-main);
        }

        #MainMenu,
        footer,
        header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }

        section[data-testid="stSidebar"] {
            background: var(--bg-sidebar);
            border-right: 1px solid #2f3136;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: var(--text-main);
        }

        .block-container {
            max-width: 900px;
            padding-top: 1.2rem;
            padding-bottom: 6rem;
        }

        .app-title {
            font-size: 1.3rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: var(--text-muted);
            margin-bottom: 1.25rem;
            font-size: 0.95rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 0.5rem;
            padding: 0.85rem 1rem;
            margin: 0 0 0.8rem 0;
            border: 1px solid transparent;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background: var(--bg-assistant);
            border-color: var(--border-color);
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background: transparent;
            border-color: #4a4c5a;
        }

        div[data-testid="stChatMessageContent"] p {
            font-size: 0.98rem;
            line-height: 1.6;
            color: var(--text-main);
        }

        div[data-testid="stChatInput"] textarea {
            background: #40414f;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            border-radius: 0.85rem;
        }

        div[data-testid="stChatInput"] textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }

        .metadata-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _call_backend(question: str) -> Dict[str, Any]:
    payload = {
        "question": question,
        "top_k": int(st.session_state["top_k"]),
        "mode": str(st.session_state["mode"]),
        "reasoning": bool(st.session_state["reasoning"]),
    }

    response = requests.post(
        f"{st.session_state['backend_url'].rstrip('/')}/ask",
        json=payload,
        timeout=int(st.session_state["request_timeout_s"]),
    )
    response.raise_for_status()
    data: Dict[str, Any] = response.json()
    if data.get("status") != "success":
        raise RuntimeError(str(data.get("message", "Backend returned an error status")))
    return data


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Conversation")
        st.button("New chat", use_container_width=True, on_click=_reset_chat)
        st.divider()

        st.markdown("### Connection")
        st.text_input("Backend URL", key="backend_url")
        st.number_input(
            "Request timeout (seconds)",
            min_value=10,
            max_value=600,
            step=5,
            key="request_timeout_s",
        )
        st.divider()

        st.markdown("### Retrieval")
        st.selectbox("Mode", options=["hybrid", "local", "online"], key="mode")
        st.slider("Top K", min_value=1, max_value=10, step=1, key="top_k")
        st.toggle("Reasoning mode", key="reasoning")
        st.toggle("Show metadata", key="show_metadata")


def _render_header() -> None:
    st.markdown('<div class="app-title">RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Chat interface optimized for local retrieval and generation.</div>',
        unsafe_allow_html=True,
    )


def _render_metadata(meta: Dict[str, Any]) -> None:
    citations: List[Dict[str, Any]] = list(meta.get("citations", []))
    sources: List[Dict[str, Any]] = list(meta.get("sources", []))
    processing_time = meta.get("processing_time")
    mode = meta.get("mode")
    confidence = meta.get("confidence")

    if not any(
        [citations, sources, processing_time is not None, mode, confidence is not None]
    ):
        return

    with st.expander("Response details", expanded=False):
        if processing_time is not None:
            st.markdown(f"- Processing time: `{float(processing_time):.2f}s`")
        if mode:
            st.markdown(f"- Mode: `{mode}`")
        if confidence is not None:
            st.markdown(f"- Confidence: `{float(confidence):.3f}`")
        if citations:
            st.markdown("Citations:")
            for item in citations[:8]:
                source = item.get("source", "unknown")
                chunk_id = item.get("chunk_id", "unknown")
                st.markdown(f"- `{source}:{chunk_id}`")
        if sources:
            st.markdown("Sources:")
            for item in sources[:5]:
                source = item.get("source", "unknown")
                chunk_id = item.get("chunk_id", "unknown")
                st.markdown(f"- `{source}` / `{chunk_id}`")


def _render_messages() -> None:
    for message in st.session_state["messages"]:
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and st.session_state["show_metadata"]:
                metadata = message.get("meta", {})
                if isinstance(metadata, dict):
                    _render_metadata(metadata)


def _handle_prompt(prompt: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    try:
        with st.spinner("Generating response..."):
            data = _call_backend(prompt)

        answer = str(data.get("answer", "")).strip() or "No answer returned by backend."
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "meta": {
                    "processing_time": data.get("processing_time"),
                    "mode": data.get("mode"),
                    "confidence": data.get("confidence"),
                    "citations": data.get("citations", []),
                    "sources": data.get("sources", []),
                },
            }
        )
    except Exception as exc:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"Request failed: {exc}",
                "meta": {"error": True},
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="RAG Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _inject_styles()
    _render_sidebar()
    _render_header()
    _render_messages()

    prompt = st.chat_input("Send a message")
    if prompt:
        _handle_prompt(prompt.strip())
        st.rerun()


if __name__ == "__main__":
    main()
