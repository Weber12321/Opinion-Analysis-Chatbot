from langchain_core.messages import HumanMessage
from app.agents.opinion_analysis_workflow import (
    OpinionAnalysisWorkflow,
)

if __name__ == "__main__":

    workflow = OpinionAnalysisWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="關於蔡英文的新聞。")],
        "is_search_related": False,
        "search_results": [],
        "analysis_results": {},
    }

    response = workflow.workflow.invoke(
        initial_state,
        config={"configurable": {"thread_id": "1"}},
    )
    print(response["messages"][-1])
