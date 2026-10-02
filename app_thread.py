import os
import uuid
import tempfile

import streamlit as st

from Chatbot import (
    app,
    get_threads,
    ingest_rag_document
)

from langchain_core.messages import HumanMessage


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cognivault",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# GENERATE NEW THREAD ID
# =========================================================

def generate_uuid():

    return str(
        uuid.uuid4()
    )


# =========================================================
# ADD THREAD
# =========================================================

def add_thread(thread_id):

    if thread_id not in st.session_state[
        "chat_threads"
    ]:

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


# =========================================================
# CREATE NEW CHAT
# =========================================================

def reset_chat():

    new_thread = generate_uuid()

    st.session_state[
        "thread_id"
    ] = new_thread

    st.session_state[
        "message_history"
    ] = []

    # A new chat starts without an active document.
    st.session_state[
        "active_document_id"
    ] = None

    st.session_state[
        "active_document_name"
    ] = None

    add_thread(
        new_thread
    )


# =========================================================
# LOAD OLD CHAT
# =========================================================

def load_chat(thread_id):

    st.session_state[
        "thread_id"
    ] = thread_id

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = app.get_state(
        config
    )

    messages = state.values.get(
        "messages",
        []
    )

    # Restore the document selected for this conversation.
    st.session_state[
        "active_document_id"
    ] = state.values.get(
        "active_document_id"
    )

    st.session_state[
        "active_document_name"
    ] = state.values.get(
        "active_document_name"
    )

    st.session_state[
        "message_history"
    ] = []

    for message in messages:

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        if message.type == "human":

            if message.content:

                st.session_state[
                    "message_history"
                ].append({

                    "role": "user",

                    "content": message.content

                })

        # -------------------------------------------------
        # FINAL AI MESSAGE
        # -------------------------------------------------

        elif (

            message.type == "ai"

            and message.content

            and not message.tool_calls

        ):

            st.session_state[
                "message_history"
            ].append({

                "role": "assistant",

                "content": message.content

            })


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

if "message_history" not in st.session_state:

    st.session_state[
        "message_history"
    ] = []


if "active_document_id" not in st.session_state:

    st.session_state[
        "active_document_id"
    ] = None


if "active_document_name" not in st.session_state:

    st.session_state[
        "active_document_name"
    ] = None


if "chat_threads" not in st.session_state:

    st.session_state[
        "chat_threads"
    ] = get_threads()


if "thread_id" not in st.session_state:

    new_thread = generate_uuid()

    st.session_state[
        "thread_id"
    ] = new_thread

    add_thread(
        new_thread
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "💬 My Conversations"
)


# =========================================================
# NEW CHAT
# =========================================================

if st.sidebar.button(

    "➕ New Chat",

    use_container_width=True

):

    reset_chat()

    st.rerun()


st.sidebar.divider()


# =========================================================
# OLD CHAT LIST
# =========================================================

st.sidebar.subheader(
    "Previous Chats"
)


for thread_id in st.session_state[
    "chat_threads"
]:

    config = {

        "configurable": {

            "thread_id": thread_id

        }

    }

    state = app.get_state(
        config
    )

    messages = state.values.get(
        "messages",
        []
    )

    title = "New Chat"


    # -----------------------------------------------------
    # Find first human message
    # -----------------------------------------------------

    for message in messages:

        if message.type == "human":

            if message.content:

                title = message.content[:30]

                if len(
                    message.content
                ) > 30:

                    title += "..."

            break


    # -----------------------------------------------------
    # Chat button
    # -----------------------------------------------------

    if st.sidebar.button(

        title,

        key=f"thread_{thread_id}",

        use_container_width=True

    ):

        load_chat(
            thread_id
        )

        st.rerun()


# =========================================================
# MAIN TITLE
# =========================================================

col1, col2, col3 = st.columns(
    [1, 2, 1]
)

with col2:

    st.title(
        "Cognivault"
    )

    if st.session_state.get("active_document_name"):
        st.caption(
            f"📄 Active document: "
            f"{st.session_state['active_document_name']}"
        )


# =========================================================
# CURRENT CHAT HISTORY
# =========================================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT + FILE UPLOAD
# =========================================================

chat_input = st.chat_input(

    "Ask me anything...",

    accept_file=True,

    file_type=["pdf"]

)


# =========================================================
# HANDLE USER INPUT
# =========================================================

if chat_input:

    # -----------------------------------------------------
    # GET TEXT
    # -----------------------------------------------------

    user_input = chat_input.text


    # -----------------------------------------------------
    # GET UPLOADED FILES
    # -----------------------------------------------------

    uploaded_files = chat_input.files


    # -----------------------------------------------------
    # PROCESS UPLOADED PDF
    # -----------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            temp_path = None

            try:

                # -----------------------------------------
                # Save PDF temporarily
                # -----------------------------------------

                with tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=".pdf"

                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                # -----------------------------------------
                # Send PDF to RAG pipeline
                # -----------------------------------------

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    document_info = ingest_rag_document(
                        temp_path
                    )

                # Keep the returned document ID. The backend uses
                # this ID to restrict retrieval to this document.
                st.session_state[
                    "active_document_id"
                ] = document_info["document_id"]

                st.session_state[
                    "active_document_name"
                ] = document_info["source"]

                st.toast(
                    f"📄 {uploaded_file.name} added "
                    f"({document_info['chunks']} chunks)"
                )


            except Exception as e:

                st.error(
                    f"Error processing document: {e}"
                )


            finally:

                # -----------------------------------------
                # Delete temporary file
                # -----------------------------------------

                if (

                    temp_path is not None

                    and os.path.exists(temp_path)

                ):

                    os.remove(
                        temp_path
                    )


    # -----------------------------------------------------
    # ONLY CONTINUE IF USER ENTERED TEXT
    # -----------------------------------------------------

    if user_input:

        # =================================================
        # DISPLAY USER MESSAGE
        # =================================================

        with st.chat_message("user"):

            st.markdown(
                user_input
            )


        # =================================================
        # STORE USER MESSAGE LOCALLY
        # =================================================

        st.session_state[
            "message_history"
        ].append({

            "role": "user",

            "content": user_input

        })


        # =================================================
        # LANGGRAPH CONFIG
        # =================================================

        config = {

            "configurable": {

                "thread_id": (
                    st.session_state[
                        "thread_id"
                    ]
                )

            }

        }


        # =================================================
        # CALL LANGGRAPH
        # =================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    result = app.invoke(

                        {

                            "messages": [

                                HumanMessage(
                                    content=user_input
                                )

                            ],

                            # This is required by the corrected backend
                            # so rag_tool retrieves only this document.
                            "active_document_id": (
                                st.session_state[
                                    "active_document_id"
                                ]
                            )

                        },

                        config=config

                    )


                    # -------------------------------------
                    # Get final AI message
                    # -------------------------------------

                    ai_message = (
                        result["messages"][-1]
                    )

                    ai_response = (
                        ai_message.content
                    )


                    # -------------------------------------
                    # Display response
                    # -------------------------------------

                    st.markdown(
                        ai_response
                    )


                    # -------------------------------------
                    # Store response
                    # -------------------------------------

                    st.session_state[
                        "message_history"
                    ].append({

                        "role": "assistant",

                        "content": ai_response

                    })


                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )