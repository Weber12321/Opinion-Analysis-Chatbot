import uuid
import streamlit as st
import os
from typing import Dict, List, Tuple, Optional


# Page configuration
st.set_page_config(
    page_title="Opinion Analysis Chatbot",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Application header
st.title("Opinion Analysis Chatbot")
st.markdown("""
    This chatbot answers questions related to opinion analysis, public sentiment, 
    and news analysis. Ask questions like:
    - What is the public sentiment about AI adoption?
    - Analyze news coverage on climate change this month
    - How are people reacting to the latest tech product release?
""")

def process_history_messages(history: List[Dict]) -> str:
    """Process chat history messages for display"""
    processed_messages = ""
    for message in history:
        processed_messages += f"{message['role'].capitalize()}: {message['content']}\n\n"
    return processed_messages

# Function to interact with opinion_analysis API
def chat_with_api(message: str) -> Tuple[str, Optional[str]]:
    """Send message to opinion_analysis API and get response"""
    try:
        payload = {
            "message": message,
            "conversation_id": st.session_state.conversation_id
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{API_URL}/chat", json=payload)
            
        if response.status_code == 200:
            data = response.json()
            return data["response"], data["conversation_id"]
        else:
            return f"Error: {response.status_code} - {response.text}", None
    except Exception as e:
        return f"Error communicating with opinion_analysis: {str(e)}", None

# Function to format entity display
def format_entity(entity_data: Dict) -> str:
    """Format entity data for display"""
    return f"{entity_data['text']} ({entity_data['label']})"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about opinion analysis..."):
    # Add user message to chat history

    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = str(uuid.uuid4())

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing opinions..."):
            response_text, conversation_id = chat_with_api(prompt)
            
            # Update conversation ID
            if conversation_id:
                st.session_state.conversation_id = conversation_id
            
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
        st.session_state.conversation_id = None
        st.rerun()
