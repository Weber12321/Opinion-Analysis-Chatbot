import uuid
import streamlit as st
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.self_rag_workflow import (
    SelfRAGWorkflow,
)

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📊",
    layout="wide",
)


# Initialize the workflow
@st.cache_resource
def get_workflow():
    return SelfRAGWorkflow()


workflow = get_workflow()

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Application header
st.title("RAG Chatbot")
st.markdown(
    """
    歡迎來到 RAG 機器人！這個聊天機器人專注於回答組織內文檔的問題。您可以詢問有關以下主題的問題：
    - KEYPO 的「熱門關鍵字」是如何計算出來的？
    - 使用 KEYPO 的「警報信」功能時，使用者可以自訂哪些設定？
    - KEYPO 的「GPT 報告」API 主要包含哪些分析面向？
      
    請注意，若查詢與文檔無關內文機器人將不會進一步回答。
    """
)


def convert_to_langchain_messages(messages: List[Dict]) -> List:
    """Convert the session state messages to langchain message objects"""
    lc_messages = []
    for message in messages:
        if message["role"] == "user":
            lc_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            lc_messages.append(AIMessage(content=message["content"]))
    return lc_messages


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about opinion analysis..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process message through workflow
    with st.chat_message("assistant"):
        # Convert stored messages to langchain format
        lc_messages = convert_to_langchain_messages(st.session_state.messages)

        # Add the new user message
        lc_messages.append(HumanMessage(content=prompt))

        # Create thread_id if not exists
        thread_id = st.session_state.thread_id
        if not thread_id:
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id

        # Prepare initial state for the workflow
        initial_state = {"messages": lc_messages, "max_generation": 2}
        # Process through workflow and get response
        with st.spinner("Processing your query..."):
            # Create progress bars for each step
            search_progress = st.empty()
            analysis_progress = st.empty()

            # Run first step - searching
            search_progress.text("Searching for relevant news...")

            response = workflow.workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            response_text = response["messages"][-1].content
            search_progress.text("✅ Search completed")
            st.markdown(response_text)

        # Update thread ID
        st.session_state.thread_id = thread_id

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        """
這個機器人專門根據組織內文檔做查詢以及回覆。
它可以幫助你解決針對組織內文檔內容問題解惑。
    """
    )

    st.header("Features")
    st.markdown(
        """
        🔍 向量查詢與問題改寫
        """
    )

    # Clear chat button
    if st.button("清除對話"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()
