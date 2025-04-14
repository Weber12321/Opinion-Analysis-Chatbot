from langchain_core.messages import HumanMessage
from app.agents.opinion_analysis_workflow import (
    OpinionAnalysisWorkflow,
)
from app.agents.self_rag_workflow import SelfRAGWorkflow

# if __name__ == "__main__":

#     workflow = SelfRAGWorkflow()

#     initial_state = {
#         "messages": [HumanMessage(content="關於蔡英文的新聞。")],
#         "is_retrieval_related" = False,
#         validated_docs: List[str]  # Documents that passed validation
#         response: str = ""  # Generated response
#         response_validated: Optional[bool] = None  # Whether response passed validation
#         max_generation: int = 2  # Maximum number of retries
#         query_rewritten: bool = False  # Whether query was rewritten
#         rewritten_query: str = ""  # Rewritten query if any
#     }

#     response = workflow.workflow.invoke(
#         initial_state,
#         config={"configurable": {"thread_id": "1"}},
#     )
#     print(response["messages"][-1])
