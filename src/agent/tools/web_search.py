# src/agent/tools/web_search.py
import os
import time
from dotenv import load_dotenv
from typing import Dict, Optional, List, Union
from tavily import TavilyClient
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import concurrent.futures

load_dotenv()

# ===== Load environment variables =====
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
MAX_SEARCH_THREADS = int(os.environ.get("MAX_SEARCH_THREADS", 3)) 
MAX_SEARCH_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", 3)) 
TAVILY_CLIENT = TavilyClient(api_key=TAVILY_API_KEY)

# ===== Define the structure of the search input =====
class SearchInput(BaseModel):
    """
    Input schema for web search operations. 
    Queries parameter only accepts English keywords.
    
    Attributes:
        queries (list[str]): The search query strings that will be used to perform the web search.
    """
    
    queries: List[str] = Field(
        title="Queries",
        description=(
            "A list of query strings for web search."
            "Each query must be in English and should be specific enough to yield relevant results."
            "For best results, use clear and well-formed questions or keyword combinations."
        ),
    )

# ===== Define the Tavily search class =====
class TavilySearch:
    def __init__(self):
        pass

    def search(self, query: str) -> Dict:
        try:
            return TAVILY_CLIENT.search(
                query,
                max_results=MAX_SEARCH_RESULTS,
            )
        except Exception as e:
            print(f"Error in Tavily search: {str(e)}")
            raise e

    def get_search_results(self, queries: Union[str, List[str]]) -> Dict[str, Union[List[Dict], str]]:
        """
        Execute search and return both structured list and LLM-formatted text results
        
        Args:
            queries: Single query string or list of queries
            
        Returns:
            Dict containing:
                - 'results': List of result dictionaries for DataFrame conversion
                - 'llm_text': Formatted text for LLM input
        """
        if isinstance(queries, str):
            queries = [queries]

        results_for_df = []
        llm_formatted = []

        def search_and_process(query: str) -> tuple[list, str]:
            print(f"Searching for: {query}")
            start_time = time.time()
            
            results = self.search(query)
            df_results = []
            text_parts = [f"Results for: {query}"]
            
            for result in results.get('results', []):
                # Make sure the result has the necessary fields
                df_results.append({
                    'query': query,
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0.0),
                    'published_date': result.get('published_date', '')
                })
                
                # Add the result details to the LLM-formatted text
                text_parts.extend([
                    f"title: {result.get('title')}",
                    f"description: {result.get('content')}",
                    f"url: {result.get('url')}",
                    ""
                ])
            
            return df_results, "\n".join(text_parts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
            for df_items, text_result in executor.map(search_and_process, queries):
                results_for_df.extend(df_items)
                llm_formatted.append(text_result)

        return {
            'results': results_for_df,
            'llm_text': "\n\n".join(llm_formatted)
        }

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.get_search_results,
            name="tavily_web_search",
            description=self.__doc__,
            args_schema=SearchInput,
            return_direct=True
        )


if __name__ == "__main__":
    try:
        print("Starting Web Search test...")
        search = TavilySearch()
        test_queries = ["Python programming language news", "Latest AI advancements"]
        results = search.get_search_results(test_queries)
        
        # Test conversion to DataFrame
        import pandas as pd
        df = pd.DataFrame(results['results'])
        print("\nDataFrame Preview:")
        print(df.head())
        print("\nDataFrame Columns:", df.columns.tolist())
        
        # Test LLM-formatted text
        print("\nLLM Formatted Text:")
        print(results['llm_text'])
        
        print("\nTest completed: Success ✅")
        
    except Exception as e:
        print(f"\nTest failed ❌: {str(e)}")