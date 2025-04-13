import uuid
import streamlit as st
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from opinion_analysis.app.agents.opinion_analysis_workflow import OpinionAnalysisWorkflow

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
st.markdown("""
    This chatbot answers questions related to opinion analysis, public sentiment, 
    and news analysis. Ask questions like:
    - What is the public sentiment about AI adoption?
    - Analyze news coverage on climate change this month
    - How are people reacting to the latest tech product release?
""")

def convert_to_langchain_messages(messages: List[Dict]) -> List:
    """Convert the session state messages to langchain message objects"""
    lc_messages = []
    for message in messages:
        if message["role"] == "user":
            lc_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            lc_messages.append(AIMessage(content=message["content"]))
    return lc_messages

def process_message_with_workflow(message: str) -> Tuple[str, str]:
    """Process a message using the OpinionAnalysisWorkflow"""
    # Convert stored messages to langchain format
    lc_messages = convert_to_langchain_messages(st.session_state.messages)
    
    # Add the new user message
    lc_messages.append(HumanMessage(content=message))
    
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
        "analysis_results": {}
    }
    
    # Process through workflow and get response
    with st.spinner("Processing your query..."):
        # Create progress bars for each step
        search_progress = st.empty()
        analysis_progress = st.empty()
        
        # Run first step - searching
        search_progress.text("Searching for relevant news...")
        thread = workflow.workflow.with_thread_id(thread_id)
        result = thread.invoke(initial_state)
        
        search_progress.text("✅ Search completed")
        
        # Extract response
        if result["messages"] and isinstance(result["messages"][-1], AIMessage):
            response_text = result["messages"][-1].content
        else:
            response_text = "I couldn't generate a proper response. Please try again."
    
    return response_text, thread_id

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
        response_text, thread_id = process_message_with_workflow(prompt)
        
        # Update thread ID
        st.session_state.thread_id = thread_id
        
        # Display response
        st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
        This chatbot specializes in opinion analysis based on web news.
        It can help you understand public sentiment, track trending topics,
        and analyze how opinions evolve over time.
    """)
    
    st.header("Features")
    st.markdown("""
        - 📰 Real-time news analysis
        - 🔍 Named Entity Recognition
        - 😊 Sentiment analysis
        - 📊 Opinion trends
    """)
    
    # Clear chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()
