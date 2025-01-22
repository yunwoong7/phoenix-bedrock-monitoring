import sys
import os
import operator
from typing import Annotated
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.agents import AgentFinish
from colorama import Fore, Style
from langgraph.graph import END, StateGraph, START
from src.agent.states.schemas import AgentState, Plan
from src.agent.nodes.planner import Planner
from src.agent.nodes.executor import Executor
from src.agent.nodes.responder import Responder

from src.agent.llm.llm_interface import LLMChat
from src.agent.llm.planner_model import PlannerModel
from src.agent.llm.responder_model import ResponderModel
from src.agent.tools.web_search import TavilySearch
from typing import Dict
import time
from src.monitoring.phoenix import setup_phoenix

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
        print(Fore.GREEN + "\n🔍 Checking Execution State:" + Fore.WHITE + "(Next step:" + next_step + ")" + Style.RESET_ALL)
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
        workflow.add_node("planner", Planner(self.planner_model, self.tools))
        # Add the executor node
        workflow.add_node("executor", Executor(self.planner_model, self.tools))
        # Add the response node
        workflow.add_node("respond", Responder(self.responder_model))
        
        # ========== Add edges to the workflow graph ==========
        # From start to planner
        workflow.add_edge(START, "planner")
        # # From planner to executor
        # workflow.add_edge("planner", "executor")
        # # From executor to responder
        # workflow.add_edge("executor", "respond")
        # From responder to end
        workflow.add_edge("respond", END)

        # ========== Add conditional edges to the workflow graph ==========
        workflow.add_conditional_edges(
            "planner",
            # Check if we need to execute tasks
            self.check_execution_needed,
            {
                "execute": "executor",
                "respond": "respond"
            }
        )
        
        workflow.add_conditional_edges(
            "executor",
            # Check if we need more planning
            self.check_next_step,
            {
                "plan": "planner",
                "respond": "respond"
            }
        )
        
        return workflow.compile()

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('mode', nargs='?', default='default')  # mode는 선택적 인자
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
           test_query = "hi"
       
       initial_state = AgentState.initial_state()
       initial_state.input = test_query

       # Run agent
       result = agent.workflow.invoke(initial_state)
       
       print("\n✅ Test completed successfully!")
       
   except Exception as e:
       print(f"\n❌ Test failed: {str(e)}")