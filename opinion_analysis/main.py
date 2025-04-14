import streamlit as st


pages = {
    "機器人服務": [
        st.Page(
            "page/chatbot.py",
            title="輿情分析機器人",
            icon="📊",
        ),
        st.Page(
            "page/chatbot_rag.py",
            title="向量檢索機器人",
            icon="📊",
        ),
    ]
}


pg = st.navigation(pages)
pg.run()
