from pydantic import BaseModel, Field
from typing import List, Dict
from colorama import Fore, Style 

class Task(BaseModel):
    """A single task in the plan"""
    title: str = Field(default=None, description="The title of the task")
    tool_name: str = Field(default="", description="The name of the tool to use for this task")
    tool_args: dict = Field(default={}, description="The arguments to be passed to the tool")
    description: str = Field(default="", description="The detailed description of what needs to be done")

class Plan(BaseModel):
    """Plan consists of tasks to complete a given query"""
    requires_tool: bool = Field(default=False, description="Whether the plan requires a tool to execute")
    direct_response: str = Field(default="", description="Direct response if no tool is required") 
    overview: str = Field(default="", description="Brief overview of the plan")
    tasks: List[Task] = Field(default_factory=list, description="List of tasks to execute")

    def __str__(self) -> str:
        plan_str = [
            "────────────────────────────────────────────────────",
            f"🚀 Requires Tool: {self.requires_tool}",
            f"📝 Direct Response: {self.direct_response}",
            f"📋 Overview: {self.overview}",
            "🔍 Tasks:"
        ]
        
        for i, task in enumerate(self.tasks, 1):
            plan_str.append(f"{i}. {task.title}")
            if task.tool_name:
                plan_str.append(f"   Tool: {task.tool_name}")
                plan_str.append(f"   Args: {task.tool_args}")
            plan_str.append(f"   Description: {task.description}")
            plan_str.append("")
            
        plan_str.append("────────────────────────────────────────────────────")
        return "\n".join(plan_str)

# Define the state of the agent  
class AgentState(BaseModel):
    """manage the state of the agent"""
    input: str
    plan: Plan
    executed_tasks: List[Dict] = []  # 실행된 태스크들의 결과
    remaining_tasks: List[Task] = []  # 아직 실행되지 않은 태스크들
    search_query: str = ""
    search_results: str =""
    response: str = ""
    message_history: List[Dict] = []

    def initial_state():
        return AgentState(
            input="",
            plan=Plan(
                requires_tool=False,
                direct_response="",
                overview="",
                tasks=[]
            ),
            executed_tasks=[],
            remaining_tasks=[],
            search_query="",
            search_results="",
            response="",
            message_history=[]
        )