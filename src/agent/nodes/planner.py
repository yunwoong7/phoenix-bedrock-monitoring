import sys
import os
import time
from colorama import Fore, Style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import cast
from copy import deepcopy
from datetime import datetime, timezone
import asyncio

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import StructuredTool
from src.agent.llm.llm_interface import LLMChat
from src.agent.states.schemas import AgentState, Plan
from langchain_core.messages import HumanMessage, AIMessage

SYSTEM_PROMPT = """You are an AI research assistant. Follow these steps to analyze and respond to queries:

1. First, determine if the query can be answered directly without tools:
   - Simple greetings or basic questions
   - General knowledge within your capabilities
   - Questions not requiring real-time or external data

2. If direct response is possible:
   - Respond immediately without creating a plan
   - Keep the response clear and concise

3. If tools are needed:
   - Create a focused execution plan with:
     a. Brief overview of approach
     b. Specific tool tasks with clear parameters
     c. Structured sequence of actions

Always use the provided datetime for any time-sensitive information.
""".strip()

INSTRUCTION = """<current-datetime>{datetime}</current-datetime>

Available tools: 
{tool_desc}

Query: {query}

Determine if this query:
1. Can be answered directly (respond without tools)
2. Requires tool usage (provide structured plan)

Response format:
- For direct answers: Provide the response
- For tool usage: Create a plan with specific steps""".strip()

class Planner:
    def __init__(self, llm: LLMChat, tools: list[StructuredTool]) -> None:
        self.model = llm.model.with_structured_output(Plan)
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION
        self.tool_desc = self._generate_tool_desc(tools)

    def _generate_tool_desc(self, tools: list[StructuredTool]) -> str:
        tool_descs = []
        for tool in tools:
            tool_descs.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descs)

    def _build_messages(self, state: AgentState) -> PromptValue:
        messages = []
        for msg in state.message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=state.input))

        return ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("human", self.instruction),
                ("placeholder", "{conversation}")
            ]
        ).invoke(
            {   "conversation": messages,
                "datetime": datetime.now(timezone.utc).isoformat(),
                "tool_desc": self.tool_desc,
                "query": state.input
            }
        )
    
    def __call__(self, state: AgentState) -> AgentState:
            """Synchronous execution"""
            return asyncio.run(self.acall(state))
   
    async def acall(self, state: AgentState) -> AgentState:
        """Async execution"""
        print(Fore.CYAN + "\n❓Query: " + Style.RESET_ALL, state.input)
        print(Fore.GREEN + "\n📋 Generated Plan:" + Style.RESET_ALL)

        start_time = time.time()
        # Generate plan
        response = await self.model.ainvoke(self._build_messages(state))
        result = cast(Plan, response)
        elapsed_time = time.time() - start_time

        print(Fore.WHITE + str(result) + Style.RESET_ALL)
        print(Fore.YELLOW + f"⏱️ Plan generation took: {elapsed_time:.2f} seconds" + Style.RESET_ALL)
        
        return {
            **state.model_dump(),
            "plan": result,
        }
    
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from src.agent.tools.web_search import TavilySearch
    from src.agent.llm.planner_model import PlannerModel

    planner_model = PlannerModel()
    web_search_tool = TavilySearch()
    tools = [web_search_tool.get_tool()]

    async def run_tests():
        # Initialize planner with mock LLM
        planner = Planner(planner_model, tools)
        
        # Test cases
        test_queries = [
            "What is 2+2?"
        ]

        print(Fore.WHITE + "\n=== Starting Planner Tests ===\n" + Style.RESET_ALL)
        
        for query in test_queries:
            print(Fore.WHITE + f"\n📝 Testing Query: {query}" + Style.RESET_ALL)
            print("=" * 50)
            
            # Create initial state
            state = AgentState.initial_state()
            try:
                # Execute planner
                result = await planner.acall(state)
                
                # Print results (plan is already printed in acall)
                print(Fore.WHITE + "\n🔍 Final State:" + Style.RESET_ALL)
                print(f"Input: {result['input']}")
                print(f"Plan: {result['plan']}")
                print(f"Response: {result['response']}")
            except Exception as e:
                print(Fore.RED + f"❌ Error during test: {str(e)}" + Style.RESET_ALL)

        print(Fore.WHITE + "\n=== Tests Completed ===" + Style.RESET_ALL)

    # Run the tests
    asyncio.run(run_tests())