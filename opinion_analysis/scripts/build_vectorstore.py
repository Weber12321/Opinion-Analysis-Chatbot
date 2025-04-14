import os
import glob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def check_directory_exists() -> bool:
    """Check if a directory exists.

    Args:
        directory (str): The path to the directory to check.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    directory = os.getenv("VECTORSTORE_PATH", "fixtures/vector_db")

    return os.path.exists(directory)


def main():
    """Build the vector store for markdown document retrieval and return retriever.

    Reads all markdown files from the fixtures directory and creates a vector store.
    Ignores README.md files.

    Returns:
        A vector store retriever for document search
    """
    # Configure the header splitter for markdown documents

    if check_directory_exists():
        print("Directory already exists. Exiting...")
        return

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

    # Find all markdown files in the fixtures directory
    fixtures_dir = os.getenv("DOCUMENTS_PATH", "fixtures")

    assert os.path.exists(
        fixtures_dir
    ), f"Fixtures directory does not exist: {fixtures_dir}"

    markdown_files = glob.glob(os.path.join(fixtures_dir, "**/*.md"), recursive=True)

    # Filter out README.md files
    markdown_files = [
        f for f in markdown_files if os.path.basename(f).lower() != "readme.md"
    ]

    if not markdown_files:
        raise FileNotFoundError(
            f"No markdown files found in the fixtures directory: {fixtures_dir}"
        )

    # Process each markdown file
    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()

            # Get file name from the file path
            file_name = os.path.basename(file_path)

            # Process markdown files
            md_header_splits = markdown_splitter.split_text(file_content)

            # Add file metadata to each split
            for split in md_header_splits:
                split.metadata["source"] = file_path
                split.metadata["document"] = file_name

            all_splits.extend(md_header_splits)
            print(f"Processed: {file_path}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # If no documents were processed, raise an error
    if not all_splits:
        raise ValueError("No documents were processed from the markdown files.")

    # Further split by characters
    final_splits = text_splitter.split_documents(all_splits)

    # Create vector store
    vector_store = FAISS.from_documents(final_splits, OpenAIEmbeddings())

    # Save the vector store locally
    save_path = os.getenv("VECTORSTORE_PATH", "fixtures/vector_db")
    vector_store.save_local(save_path)
    print(f"Vector store saved to {save_path}")


if __name__ == "__main__":
    main()
