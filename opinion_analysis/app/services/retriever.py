from io import BytesIO
import os
from typing import List, Union
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


@tool("retrieve", return_direct=True)
def retrieve(query: str) -> Union[List[str] | str]:
    """Retrieve relevant documents from the vector store based on the query.
    Args:
        query (str): The search query.
    Returns:
        List[str]: A list of relevant document contents.
    """
    vectorstore = FAISS.load_local(
        os.getenv("VECTORSTORE_PATH", "fixtures/vector_db"), OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    return [doc.page_content for doc in docs]
