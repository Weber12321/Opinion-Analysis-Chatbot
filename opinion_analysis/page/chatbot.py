import time
import uuid
import streamlit as st
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.opinion_analysis_workflow import (
    OpinionAnalysisWorkflow,
)

# Page configuration
st.set_page_config(
    page_title="Opinion Analysis Chatbot",
    page_icon="📊",
    layout="wide",
)


# Initialize the workflow
@st.cache_resource
def get_workflow():
    return OpinionAnalysisWorkflow()


workflow = get_workflow()

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Application header
st.title("Opinion Analysis Chatbot")
st.markdown(
    """
    This chatbot answers questions related to news with opinion analysis, public sentiment, 
    and semantic like name entity, sentiment analysis. Ask questions like:
    - Can you search a news about AI adoption?
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
        initial_state = {
            "messages": lc_messages,
            "is_search_related": False,
            "search_results": [],
            "analysis_results": {},
        }
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
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        """
        This chatbot specializes in opinion analysis based on web news.
        It can help you understand public sentiment, track trending topics,
        and analyze how opinions evolve over time.
    """
    )

    st.header("Features")
    st.markdown(
        """
        - 📰 Real-time news analysis
        - 🔍 Named Entity Recognition
        - 😊 Sentiment analysis
        - 📊 Opinion trends
    """
    )

    # Clear chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()
