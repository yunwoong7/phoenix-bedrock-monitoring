# src/agent/workflow.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from colorama import Fore, Style
from langgraph.graph import END, StateGraph
from src.agent.states.schemas import AgentState
from src.agent.nodes.planner import Planner
from src.agent.nodes.executor import Executor
from src.agent.nodes.responder import Responder
from src.agent.llm.responder_model import ResponderModel
from src.agent.llm.planner_model import PlannerModel
from src.agent.tools.web_search import TavilySearch
from typing import Dict
import time
from src.monitoring.phoenix import setup_phoenix


# ===== Define the Search Agent Workflow class =====
class SearchAgentWorkflow:
    def __init__(self, tracer_provider=None):
        self.planner_model = PlannerModel(tracer_provider)
        self.responder_model = ResponderModel(tracer_provider)
        self.web_search_tool = TavilySearch()
        self.tools = [self.web_search_tool.get_tool()]
        self.workflow = self._create_workflow()

    def check_execution_needed(self, state: AgentState) -> str:
        """Check if we need to execute tasks"""
        start_time = time.time()
        # check if we need to execute tasks
        next_step = "execute" if state.plan.requires_tool and len(state.plan.tasks) > 0 else "respond"
        
        elapsed_time = time.time() - start_time
        print(Fore.GREEN + "\n🔍 Checking Execution State" + Fore.WHITE + "(Next step:" + next_step + ")" + Style.RESET_ALL)
        print(Fore.YELLOW + f"⏱️ Check took: {elapsed_time:.2f} seconds" + Style.RESET_ALL)
        return next_step

    def check_next_step(self, state: AgentState) -> str:
        """Check if we need more planning or can proceed to response"""
        print("\n🔄 Checking remaining tasks...")
        start_time = time.time()
        
        # Check if we have more tasks to execute
        next_step = "plan" if state.remaining_tasks else "respond"
        
        elapsed_time = time.time() - start_time
        print(f"Next step: {next_step}")
        print(f"⏱️ Check took: {elapsed_time:.2f} seconds")
        return next_step

    def _create_workflow(self) -> StateGraph:
        """Create workflow"""
        workflow = StateGraph(AgentState)

        # ========== Add nodes to the workflow graph ==========
        # Add the plan node
        workflow.add_node("planner", Planner(self.planner_model, self.tools, verbose=True))
        # Add the executor node
        workflow.add_node("executor", Executor(self.planner_model, self.tools))
        # Add the responder node
        workflow.add_node("responder", Responder(self.responder_model))   
        
        # ========== Add edges to the workflow graph ==========
        # From start to planner
        workflow.set_entry_point("planner")
        # From responder to end
        workflow.add_edge("responder", END)

        # ========== Add conditional edges to the workflow graph ==========
        workflow.add_conditional_edges(
            "planner",
            # Check if we need to execute tasks
            self.check_execution_needed,
            {
                "execute": "executor",
                "respond": "responder"
            }
        )
        
        workflow.add_conditional_edges(
            "executor",
            # Check if we need more planning
            self.check_next_step,
            {
                "plan": "planner",
                "respond": "responder"
            }
        )
        
        return workflow.compile()
    
    async def astream(self, state: AgentState):
        """Async stream of the workflow execution"""
        async for event in self.workflow.astream(state, stream_mode=["messages", "updates"]):
            # print(event)
            if isinstance(event, tuple):
                mode = event[0] # messages or updates
                if mode == "messages":
                    msg_chunk = event[1][0]
                    node = event[1][1].get("langgraph_node")
                    if (hasattr(msg_chunk, 'content') and 
                        msg_chunk.content and 
                        isinstance(msg_chunk.content, list) and 
                        len(msg_chunk.content) > 0):
                        content = msg_chunk.content[0]
                        if isinstance(content, dict) and content.get("type") == "tool_use":
                            token = content.get("input", "")
                            if token:
                                # yield token
                                # case for tool_use response
                                yield {
                                    node: {"token": token},
                                    "status": "in_progress"
                                }
                        elif isinstance(content, dict) and content.get("type") == "text":
                            token = content.get("text", "")
                            if token:
                                # yield token
                                # case for text response
                                yield {
                                    node: {"token": token},
                                    "status": "in_progress"
                                }
                elif mode == "updates":
                    updates = event[1]  # updates
                    yield updates


if __name__ == "__main__":
    # Test the Search Agent
    # python src/agent/workflow.py test
    import argparse
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='default')
    args = parser.parse_args()

    try:
        print("🚀 Starting Search Agent test...")
        # Phoenix 설정
        tracer_provider = setup_phoenix()
        # Create agent
        agent = SearchAgentWorkflow(tracer_provider=tracer_provider)

        if args.mode == 'test':
            # Get user input
            test_query = input("\n❓ Enter your query: ")
        else:
            # Use default test query
            test_query = "Search for the best restaurants in Seoul"
            # test_query = "Hi, how are you?"
        
        initial_state = AgentState.initial_state()
        initial_state.input = test_query

        # Run agent
        async def test_streaming():
            streaming_response = False
            async for event in agent.astream(initial_state):
                if "planner" in event:
                    if "in_progress" in event.get("status", ""):
                        token = event.get("planner", {}).get("token")
                        print(Fore.WHITE + token + Style.RESET_ALL, end='', flush=True)
                    else:
                        plan = event["planner"].get("plan")
                        print (Fore.WHITE + f"\n✅ Plan generated: {plan}" + Style.RESET_ALL)
                elif "executor" in event:
                    pass
                elif "responder" in event:
                    if "in_progress" in event.get("status", ""):
                        token = event.get("responder", {}).get("token")
                        print(Fore.WHITE + token + Style.RESET_ALL, end='', flush=True)
                        streaming_response = True
                    else:
                        if not streaming_response:
                            print(Fore.WHITE + event["responder"].get("response", "") + Style.RESET_ALL)
        asyncio.run(test_streaming())
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
