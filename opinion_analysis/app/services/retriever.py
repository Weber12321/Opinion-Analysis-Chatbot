from io import BytesIO
from typing import List, Optional
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document


def build_retriever(top_k: int = 5, uploaded_files: Optional[List[BytesIO]] = None):

    """Build the vector store for markdown document retrieval and return retriever.

    Args:
        uploaded_files: Optional list of BytesIO objects from Streamlit file uploads
                        Each file should have 'name' attribute for file name
    Returns:
        A vector store retriever for document search
    """
    # Configure the header splitter for markdown documents
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    # Set up character splitter for chunking
    chunk_size = 600
    chunk_overlap = 180
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    all_splits = []

    # Process uploaded files if provided
    if uploaded_files and len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            try:
                # Read the file content from BytesIO
                file_content = uploaded_file.getvalue().decode("utf-8")

                # Get file name from the uploaded file
                file_name = getattr(uploaded_file, "name", "unnamed_document")

                # Process based on file type
                if file_name.endswith(".md"):
                    # Process as markdown
                    md_header_splits = markdown_splitter.split_text(file_content)

                    # Add file metadata to each split
                    for split in md_header_splits:
                        split.metadata["source"] = "uploaded_file"
                        split.metadata["document"] = file_name

                    all_splits.extend(md_header_splits)
                else:
                    # Process as plain text
                    doc = Document(
                        page_content=file_content,
                        metadata={"source": "uploaded_file", "document": file_name},
                    )
                    all_splits.append(doc)

            except Exception as e:
                print(
                    f"Error processing uploaded file {getattr(uploaded_file, 'name', 'unnamed')}: {e}"
                )
                continue
    else:
        raise FileNotFoundError(
            "No uploaded files provided for vector store creation."
        )

    # If no documents were processed, create a default empty document
    if not all_splits:
        raise ValueError("No documents were processed from the uploaded files.")
    # Further split by characters if we have documents

    final_splits = text_splitter.split_documents(all_splits)
    # Create vector store
    vector_store = FAISS.from_documents(final_splits, OpenAIEmbeddings())
    
    # Save the vector store locally
    import os
    save_path = os.path.join(os.path.dirname(__file__), "../../../vector_store")
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"Vector store saved to {save_path}")

    return vector_store.as_retriever(search_kwargs={"k": top_k})


@tool("retrieve", return_direct=True)
def retrieve(query: str):
