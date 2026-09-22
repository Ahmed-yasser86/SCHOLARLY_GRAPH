from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document

from Retrival_Pipline.Graph.state import GraphState

load_dotenv()


def _academic_search_tool():
    try:
        from langchain_tavily import TavilySearch

        return TavilySearch(max_results=3)
    except Exception:
        return None


web_search_tool = _academic_search_tool()

def websearch(state: GraphState):
    question = state["question"]
    documents = state.get("documents")

    if web_search_tool is None:
        raise RuntimeError(
            "Web search is unavailable: install the optional search backend "
            "and configure its API key before using websearch."
        )

    tavily_response = web_search_tool.invoke({"query": question})
    
    if isinstance(tavily_response, dict) and "results" in tavily_response:
        results = tavily_response["results"]
    else:
        results = tavily_response

    joined_tavily_results = "\n".join([res["content"] for res in results])
    
    web_results = Document(page_content=joined_tavily_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        
    return {"documents": documents, "question": question}
