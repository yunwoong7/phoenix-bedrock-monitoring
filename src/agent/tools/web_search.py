# src/agent/tools/web_search.py
import os
import time
from dotenv import load_dotenv
from typing import Dict, Optional, List, Union
from tavily import TavilyClient
import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import concurrent.futures

load_dotenv()

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
MAX_SEARCH_THREADS = int(os.environ.get("MAX_SEARCH_THREADS", 3))  # Convert to integer
MAX_SEARCH_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", 3))  # Convert to integer

TAVILY_CLIENT = TavilyClient(api_key=TAVILY_API_KEY)

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

# Brave search tool
class BraveSearch:
    """
    Searches given queries on the web and returns the search results.

    ## Tool Parameters
    - queries (list[str]): The search query strings that will be used to perform the web search.

    ### Query generation instructions
    For each element of the queries parameter, please follow the guidelines below:
     - Generate 2-3 relevant search queries based on a given task to web searches.
     - Each query should capture the main intent of the task in different perspectives or contexts.
    """
    def __init__(self):
        self.api_key = BRAVE_API_KEY
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
    def search(self, query: str, count: Optional[int] = MAX_SEARCH_RESULTS) -> Dict:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": count
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error in Brave search: {str(e)}")
            raise e

    def get_search_results(self, queries: Union[str, List[str]], results: Optional[Dict] = None) -> str:
        if isinstance(queries, str):
            queries = [queries]
        
        formatted_results = []

        for query in queries:
            time.sleep(1)  # Add delay to avoid rate limiting
            search_results = self.search(query)
            for result in search_results.get('web', {}).get('results', []):
                formatted_results.append(
                    f"Title: {result.get('title')}\n"
                    f"Description: {result.get('description')}\n"
                    f"URL: {result.get('url')}\n"
                )
            
        return "\n\n".join(formatted_results)

    def get_tool(self) -> StructuredTool:
        """Get the web search tool"""
        return StructuredTool.from_function(
            func=self.get_search_results,
            name="brave_web_search",
            description=self.__doc__,
            args_schema=SearchInput,
            return_direct=True
        )


# Tavily search tool
class TavilySearch:
    """
    Searches given queries on the web and returns the search results.

    ## Tool Parameters
    - queries (list[str]): The search query strings that will be used to perform the web search.

    ### Query generation instructions
    For each element of the queries parameter, please follow the guidelines below:
        - Generate 2-3 relevant search queries based on a given task to web searches.
        - Each query should capture the main intent of the task in different perspectives or contexts.
    """
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

    def get_search_results(self, queries: Union[str, List[str]]) -> str:
        if isinstance(queries, str):
            queries = [queries]

        all_results = []

        def search_and_format(query: str) -> str:
            print(f"Searching for: {query}")
            start_time = time.time()
            
            results = self.search(query)
            formatted = [f"Results for: {query}"]
            
            for result in results.get('results', []):
                formatted.append(
                    f"Title: {result.get('title')}\n"
                    f"Description: {result.get('content')}\n"
                    f"URL: {result.get('url')}\n"
                )
            
            elapsed_time = time.time() - start_time
            return "\n".join(formatted)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
            search_results = list(executor.map(search_and_format, queries))
            all_results.extend(search_results)

        return "\n\n".join(all_results)

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.get_search_results,
            name="tavily_web_search",
            description=self.__doc__,
            args_schema=SearchInput,
            return_direct=True
        )

if __name__ == "__main__":
    # Basic test
    try:
        print("Starting Web Search test...")
        
        # Create search instance
        search = TavilySearch()
        
        # Get tool
        search_tool = search.get_tool()
        print("\nTool name:", search_tool.name)
        print("Tool description:", search_tool.description)
        
        # Test query
        test_query = ["Python programming language news", "Latest AI advancements"]
        print(f"\nTest queries: {test_query}")
        
        # Execute search using the tool
        formatted_results = search_tool.func(queries=test_query)
        print("\nSearch results:")
        print(formatted_results)
        
        print("\nTest completed: Success ✅")
        
    except Exception as e:
        print(f"\nTest failed ❌: {str(e)}")
