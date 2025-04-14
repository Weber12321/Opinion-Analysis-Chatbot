from app.services.llm_service import LLMService
from langgraph.graph import MessagesState, StateGraph


class SelfRAGWorkflow:

    def __init__(self):
        self.retriever = self._build_vecter_retriever()
        self.workflow = self._build_workflow()
        self.llm_service = LLMService()

    def _build_vecter_retriever(self):
        """Build the vector store for markdown document retrieval and return retriever.

        Example:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        with open(file, "r") as f:
            markdown_document = f.read()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        # MD splits
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_document)

        # Char-level splits
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = 250
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        splits = text_splitter.split_documents(md_header_splits)

        # Create vector store
        ...

        """
        pass

    def _build_workflow(self):
        """Build the LangGraph workflow for the self-RAG agent.

        It should be retrieve_or_respond -> validate_docs(for each document) -> generate_response -> validate_response -> query_rewrite -> generate_final_response.
        If it fails in validate_docs, namely no doc related, it should go back to retrieve_or_respond keep retriving and skip the top_k, max retries twice and if still fails, go to query_rewrite.
        If it fails in validate_response, it should go to query_rewrited.
        """
        pass

    def retrieve_or_respond(self, state: MessagesState):
        """An agent which decide to retrieve relevant documents based on the query or reply the LLM answer directly"""
        pass
        return state

    def validate_docs(self, state: MessagesState):
        """Validate the retrieved documents is related to the query, keep the related documents and remove the unrelated documents"""
        pass
        return state

    def generate_response(self, state: MessagesState):
        """Generate a response based on the query and validated retrieved documents"""
        pass
        return state

    def validate_response(self, state: MessagesState):
        """Validate the generated response twice with LLM response and query"""
        pass
        return state

    def query_rewrite(self, state: MessagesState):
        """Rewrite the query based on the generated response if it fails validation in validate_response stage"""
        pass
        return state

    def generate_final_response(self, state: MessagesState):
        """Generate the final response based on the rewritten query.
        If query_rewrited, reply the message with the suggestion rewritten querys to ask user to rerun the workflow.
        if not query_rewrited, reply the message with the final response which generated at generate_response stage.
        """
        pass
        return state
