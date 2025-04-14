from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional, TypedDict, Annotated

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from app.services.llm_service import RAGLLMService
from langgraph.graph import MessagesState, StateGraph, END


class SelfRAGState(TypedDict):
    """State for the Self-RAG workflow"""

    messages: List[Any]  # List of chat messages
    docs: List[str] | str  # Retrieved documents
    is_retrieval_related: bool  # Whether the query is related to retrieval
    validated_docs: List[Document]  # Documents that passed validation
    response: str  # Generated response
    response_validated: bool  # Whether response passed validation
    retrieval_attempts: int  # Number of retrieval attempts
    query_rewritten: bool  # Whether query was rewritten
    rewritten_query: str  # Rewritten query if any


class SelfRAGWorkflow:

    def __init__(self, uploaded_files: Optional[List[BytesIO]] = None):
        self.retriever = self._build_vecter_retriever(uploaded_files)
        self.llm_service = RAGLLMService()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow for the self-RAG agent.

        It should be retrieve_or_respond -> validate_docs(for each document) -> generate_response -> validate_response -> query_rewrite -> generate_final_response.
        If it fails in validate_docs, namely no doc related, it should go back to retrieve_or_respond keep retriving and skip the top_k, max retries twice and if still fails, go to query_rewrite.
        If it fails in validate_response, it should go to query_rewrited.
        """
        # Create workflow
        workflow = StateGraph(SelfRAGState)

        # Add nodes
        workflow.add_node("retrieve_or_respond", self.retrieve_or_respond)
        workflow.add_node("validate_docs", self.validate_docs)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("validate_response", self.validate_response)
        workflow.add_node("query_rewrite", self.query_rewrite)
        workflow.add_node("generate_final_response", self.generate_final_response)

        # Define conditional edges

        # From retrieve_or_respond, either go to validate_docs or generate_final_response
        workflow.add_conditional_edges(
            "retrieve_or_respond",
            lambda state: (
                "validate_docs" if state["docs"] else "generate_final_response"
            ),
        )

        # From validate_docs, go back to retrieve_or_respond if no valid docs and attempts < max
        # Otherwise go to generate_response
        workflow.add_conditional_edges(
            "validate_docs",
            lambda state: (
                "retrieve_or_respond"
                if not state["validated_docs"]
                and state["retrieval_attempts"] < self.max_retrieval_attempts
                else "generate_response"
            ),
        )

        # From generate_response to validate_response
        workflow.add_edge("generate_response", "validate_response")

        # From validate_response, go to query_rewrite if validation fails, otherwise to final response
        workflow.add_conditional_edges(
            "validate_response",
            lambda state: (
                "query_rewrite"
                if not state["response_validated"]
                else "generate_final_response"
            ),
        )

        # From query_rewrite to generate_final_response
        workflow.add_edge("query_rewrite", "generate_final_response")

        # Final response leads to END
        workflow.add_edge("generate_final_response", END)

        # Set entry point
        workflow.set_entry_point("retrieve_or_respond")

        return workflow.compile()

    def retrieve_or_respond(self, state: SelfRAGState):
        """An agent which decide to retrieve relevant documents based on the query or reply the LLM answer directly"""
        # Extract the query from the latest human message
        response = self.llm_service.rag_agent.invoke(
            {"query": state["messages"][-1].content}
        )
        if isinstance(response["output"], str):
            state["is_retrieval_related"] = False
        else:
            state["is_retrieval_related"] = True
        state["docs"] = response["output"]
        return state

    def validate_docs(self, state: SelfRAGState):
        """Validate the retrieved documents is related to the query, keep the related documents and remove the unrelated documents"""
        query = state["query"]
        docs = state["docs"]
        validated_docs = []

        # Validation prompt template
        validation_template = """
        You are an AI document validator that determines if a document is relevant to a query.
        
        Query: {query}
        
        Document: {document}
        
        Is this document relevant to the query? Analyze the document's content and the query carefully.
        Respond with YES if relevant or NO if not relevant. Then provide a brief explanation of your reasoning.
        """

        validation_prompt = ChatPromptTemplate.from_template(validation_template)
        validation_chain = validation_prompt | self.llm | StrOutputParser()

        for doc in docs:
            response = validation_chain.invoke(
                {"query": query, "document": doc.page_content}
            )

            # Check if document is validated as relevant
            if response.strip().upper().startswith("YES"):
                validated_docs.append(doc)

        # Update the state with validated documents
        state["validated_docs"] = validated_docs

        return state

    def generate_response(self, state: SelfRAGState):
        """Generate a response based on the query and validated retrieved documents"""
        query = state["query"]
        validated_docs = state["validated_docs"]

        if not validated_docs:
            # No relevant documents were found, generate a response based on model knowledge
            response_template = """
            You are an AI assistant helping users with their questions.
            
            User Query: {query}
            
            No specific documentation was found for this query.
            Please provide a helpful response based on your general knowledge.
            """

            response_prompt = ChatPromptTemplate.from_template(response_template)
            response_chain = response_prompt | self.llm | StrOutputParser()
            response = response_chain.invoke({"query": query})
        else:
            # Generate response using the validated documents
            docs_content = "\n\n".join(
                [
                    f"Document {i+1}:\n{doc.page_content}"
                    for i, doc in enumerate(validated_docs)
                ]
            )

            response_template = """
            You are an AI assistant helping users with their questions.
            
            User Query: {query}
            
            Relevant Documentation:
            {docs_content}
            
            Based on the provided documentation, please give a detailed and accurate response to the query.
            Include specific information from the documentation when relevant.
            If the documentation doesn't fully answer the query, supplement with your general knowledge.
            """

            response_prompt = ChatPromptTemplate.from_template(response_template)
            response_chain = response_prompt | self.llm | StrOutputParser()
            response = response_chain.invoke(
                {"query": query, "docs_content": docs_content}
            )

        # Update the state with the generated response
        state["response"] = response

        return state

    def validate_response(self, state: SelfRAGState):
        """Validate the generated response twice with LLM response and query"""
        query = state["query"]
        response = state["response"]

        # First validation: Does the response adequately answer the query?
        validation1_template = """
        You are an AI response validator.
        
        Query: {query}
        
        Generated Response: {response}
        
        Does the response adequately answer the query? Consider:
        1. Does it address all parts of the question?
        2. Is it accurate based on the information available?
        3. Is it complete, or are there important aspects missing?
        
        Respond with YES if the response is adequate, or NO if it needs improvement.
        Then provide a brief explanation of what could be improved.
        """

        validation1_prompt = ChatPromptTemplate.from_template(validation1_template)
        validation1_chain = validation1_prompt | self.llm | StrOutputParser()
        validation1_result = validation1_chain.invoke(
            {"query": query, "response": response}
        )

        # Second validation: Is the response well-structured and clear?
        validation2_template = """
        You are an AI response validator.
        
        Query: {query}
        
        Generated Response: {response}
        
        Is the response well-structured and clear? Consider:
        1. Is it organized logically?
        2. Is the language clear and easy to understand?
        3. Does it avoid unnecessary jargon or explain technical terms when used?
        
        Respond with YES if the response is well-structured and clear, or NO if it needs improvement.
        Then provide a brief explanation of what could be improved.
        """

        validation2_prompt = ChatPromptTemplate.from_template(validation2_template)
        validation2_chain = validation2_prompt | self.llm | StrOutputParser()
        validation2_result = validation2_chain.invoke(
            {"query": query, "response": response}
        )

        # Consider the response valid only if both validations pass
        is_valid = validation1_result.strip().upper().startswith(
            "YES"
        ) and validation2_result.strip().upper().startswith("YES")

        # Update state
        state["response_validated"] = is_valid

        return state

    def query_rewrite(self, state: SelfRAGState):
        """Rewrite the query based on the generated response if it fails validation in validate_response stage"""
        original_query = state["query"]
        response = state["response"]

        # Prompt for query rewriting
        rewrite_template = """
        You are an AI query improvement specialist.
        
        Original Query: {query}
        
        Generated Response: {response}
        
        The response did not adequately answer the query. Please rewrite the query to be more specific,
        clearer, and better targeted to get the information needed. The rewritten query should address
        the weaknesses in the current response.
        
        Rewritten Query:
        """

        rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
        rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
        rewritten_query = rewrite_chain.invoke(
            {"query": original_query, "response": response}
        )

        # Update state
        state["query_rewritten"] = True
        state["rewritten_query"] = rewritten_query

        return state

    def generate_final_response(self, state: SelfRAGState):
        """Generate the final response based on the rewritten query.
        If query_rewrited, reply the message with the suggestion rewritten querys to ask user to rerun the workflow.
        if not query_rewrited, reply the message with the final response which generated at generate_response stage.
        """
        # Check if query was rewritten
        if state.get("query_rewritten", False):
            original_query = state["query"]
            rewritten_query = state["rewritten_query"]

            final_response = f"""
            I'm having trouble providing a complete answer to your question: "{original_query}"
            
            To better assist you, I suggest rephrasing your question. For example:
            
            "{rewritten_query}"
            
            This would help me provide a more accurate and helpful response.
            """

            # Add the final response as an AI message
            state["messages"].append(AIMessage(content=final_response))
        else:
            # Use the generated response
            response = state.get(
                "response", "I apologize, but I couldn't generate a proper response."
            )

            # Add the generated response as an AI message
            state["messages"].append(AIMessage(content=response))

        return state
