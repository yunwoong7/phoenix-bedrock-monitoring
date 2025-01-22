import sys
import os
import time

from colorama import Fore, Style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import cast
from datetime import datetime, timezone
import asyncio
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from src.agent.llm.llm_interface import LLMChat
from src.agent.states.schemas import AgentState, Plan
from langchain_core.messages import HumanMessage, AIMessage

SYSTEM_PROMPT = """You are an AI research assistant. Generate a comprehensive response based on the following cases:

1. For direct responses (no tools used):
   - Use the provided direct_response as is
   - Maintain the original tone and content

2. For tool-based responses:
   - Synthesize information from all executed tasks
   - Provide clear, well-structured answers
   - Include relevant details from each task result
   - Maintain a natural, conversational tone

Ensure responses are clear, concise, and directly address the original query.""".strip()

INSTRUCTION = """You are a helpful AI assistant. Generate a comprehensive response based on the search results.

Here is the user query:
{query}
    
Here are the search results to use:
{context}

Here are the source URLs to cite in [#] format

Guidelines for your response:
    1. Analyze and summarize key points from the tool results
    2. Generate a clear and informative response that:
        - Addresses the user's query
        - Uses natural language and conversational tone
        - Includes appropriate inline citations [#] when referencing sources
    3. Only include information from the provided search results
    4. Cite sources immediately after referenced information
    5. Use markdown formatting for lists and emphasis if needed
    6. Respond in the same language as the query
Please generate a comprehensive response based on the above information.

Generate an appropriate response following these guidelines:
1. For direct responses: Use the plan's direct_response
2. For tool-based responses: Synthesize all task results into a coherent answer""".strip()

class Responder:
    def __init__(self, llm: LLMChat) -> None:
        self.model = llm.model
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION

    def _build_context(self, state: AgentState) -> str:
        if state.plan.direct_response:
            return f"Direct response: {state.plan.direct_response}"
        
        context = f"Plan overview: {state.plan.overview}\n\nExecuted tasks and their results:\n"
        for task in state.executed_tasks:
            context += f"\nTask: {task['task'].title}\n"
            context += f"Result: {task['result']}\n"
        return context

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
                "query": state.input,
                "context": self._build_context(state)
            }
        )

    def __call__(self, state: AgentState) -> AgentState:
        """Synchronous execution"""
        return asyncio.run(self.acall(state))

    async def acall(self, state: AgentState) -> AgentState:
        """Async execution"""
        print(Fore.GREEN + "\n✨ Generating final response:" + Style.RESET_ALL)
        start_time = time.time()

        # If we have a direct response, use it without invoking the model
        if state.plan.direct_response:
            response = [state.plan.direct_response]  # 직접 응답을 리스트로 변환
        else:
            # Otherwise, generate streaming response based on task results
            response_chunks = []
            async for chunk in self.model.astream(self._build_messages(state)):
                chunk_text = cast(str, chunk.content)
                response_chunks.append(chunk_text)
                print(chunk_text, end="", flush=True)  # 디버깅을 위한 출력
            response = response_chunks

        elapsed_time = time.time() - start_time

        print(Fore.CYAN + "\nFinal response:" + Style.RESET_ALL)
        print(Fore.YELLOW + f"⏱️ Response generation took: {elapsed_time:.2f} seconds" + Style.RESET_ALL)

        return {
            **state.model_dump(),
            "response": response,
            "message_history": [*state.message_history, {
                "role": "assistant",
                "content": "".join(response)
            }]
        }