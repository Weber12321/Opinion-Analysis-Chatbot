
import streamlit as st


pages = {
    "prompt 管理": [
        st.Page(
            "page/chatbot.py",
            title="Opinion Analysis Chatbot",
            icon="📊",
        ),

    ]
}


pg = st.navigation(pages)
pg.run()