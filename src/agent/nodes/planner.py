# src/agent/nodes/planner.py
import sys
import os
from colorama import Fore, Style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Any, Dict, cast
from copy import deepcopy
from datetime import datetime, timezone
import asyncio

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import StructuredTool
try:
    from src.agent.llm.llm_interface import LLMChat
    from src.agent.states.schemas import AgentState, Plan
except ImportError:
    from agent.llm.llm_interface import LLMChat
    from agent.states.schemas import AgentState, Plan
from langchain_core.messages import HumanMessage, AIMessage


# ====== Define the system prompt and instruction for the Planner ======
SYSTEM_PROMPT = """You are an AI research assistant. Follow these steps to analyze and respond to queries:

1. First, determine if the query can be answered directly without tools:
    - Simple greetings or basic questions
    - General knowledge within your capabilities
    - Questions not requiring real-time or external data

2. If direct response is possible:
    - Respond immediately as the conversational agent
    - Keep the response clear, concise, and natural
    - Ensure the response is contextually appropriate and polite

3. If tools are needed:
    - Create a focused execution plan with:
      a. Brief overview of approach
      b. Specific tool tasks with clear parameters
      c. Structured sequence of actions

Always use the provided datetime for any time-sensitive information.
Ensure that all tool names and parameters are valid and correctly formatted.
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
- For tool usage: Create a plan with specific steps

Ensure that all tool names and parameters are valid and correctly formatted.
""".strip()


# ====== Define the Planner class ======
class Planner:
    def __init__(self, llm: LLMChat, tools: list[StructuredTool], verbose: bool = False) -> None:
        self.model = llm.model.with_structured_output(Plan)
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION
        self.tool_desc = self._generate_tool_desc(tools)
        self.verbose = verbose

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
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Async generator for planning tasks"""
        print(Fore.GREEN + "\n📝 Planning tasks" + Style.RESET_ALL)
        messages = self._build_messages(state)
        plan = None

        async for event in self.model.astream_events(messages, version='v2'):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content and chunk.content[0].get("type") == "tool_use":
                    input_text = chunk.content[0].get("input", "")
                    if input_text:
                        if self.verbose:
                            print(Fore.WHITE + input_text + Style.RESET_ALL, end='', flush=True)
                        yield input_text
                        yield {
                            **state.model_dump(),
                            "response": input_text,
                        }
            elif event["event"] == "on_chain_end":
                plan = event["data"]["output"]
                yield {
                    **state.model_dump(),
                    "plan": plan,
                }     
            
    async def acall(self, state: AgentState) -> AgentState:
        """Async execution with streaming events"""
        # Generate plan
        response = await self.model.ainvoke(self._build_messages(state))
        result = cast(Plan, response)
        
        # Return state with plan
        return {
            **state.model_dump(),
            "plan": result,
        }
    

# Test the Planner            
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from src.agent.tools.web_search import TavilySearch
    from src.agent.llm.planner_model import PlannerModel

    planner_model = PlannerModel()
    web_search_tool = TavilySearch()
    tools = [web_search_tool.get_tool()]

    async def run_tests():
        planner = Planner(planner_model, tools)
        test_queries = ["What is the weather like today?"]

        print("=== Starting Planner Tests ===")
        
        for query in test_queries:
            state = AgentState.initial_state()
            state.input = query
            
            try:
                print(Fore.CYAN + f"❓ Test query: {query}" + Style.RESET_ALL)
                async for result in planner(state):
                    if isinstance(result, str):
                        print(Fore.WHITE + result + Style.RESET_ALL, end='', flush=True)
                print("\n=====================================")
                print(Fore.WHITE + f"\n✅ Plan generated: {result['plan']}" + Style.RESET_ALL)
            except Exception as e:
                print(f"\n❌ Error during test: {str(e)}")

        print("\n=====================================")

    # Run the tests
    asyncio.run(run_tests())
