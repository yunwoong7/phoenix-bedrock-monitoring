# src/agent/nodes/responder.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
from colorama import Fore, Style
from typing import cast
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from src.agent.llm.llm_interface import LLMChat
from src.agent.states.schemas import AgentState
from langchain_core.messages import HumanMessage, AIMessage

# ===== Define the system prompt for the Responder =====
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


# ===== Define the Responder class =====
class Responder:
    def __init__(self, llm: LLMChat) -> None:
        self.model = llm.model
        self.system_prompt = SYSTEM_PROMPT
        # self.instruction = INSTRUCTION

    def _build_context(self, state: AgentState) -> tuple[str, list[str]]:
        if state.plan.direct_response:
            return f"Direct response: {state.plan.direct_response}", []
        
        # initialize URL list
        urls = []
        context_parts = [f"Plan overview: {state.plan.overview}\n\nExecuted tasks and their results:\n"]
        url_index = 1

        for task_info in state.executed_tasks:
            task = task_info['task']
            result = task_info.get('result', '')
            success = task_info.get('success', False)

            if not success:
                context_parts.append(f"Status: Failed")
                context_parts.append(f"Error: {result}")
                continue

            # Process the result
            if isinstance(result, dict) and 'results' in result and 'llm_text' in result:
                for item in result['results']:
                    if 'url' in item and item['url']:
                        urls.append(f"[{url_index}] {item['url']}")
                        url_index += 1
                context_parts.append(result['llm_text'])
            else:
                context_parts.append(str(result))

        return "\n".join(context_parts), list(dict.fromkeys(urls))

    def _build_messages(self, state: AgentState) -> PromptValue:
        messages = []
        for msg in state.message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=state.input))
        
        context, urls = self._build_context(state)
        url_text = chr(10).join(urls) if urls else ""
        instruction = f"""You are a helpful AI assistant. Generate a comprehensive response based on the search results.
        
        Here is the user query:
        {state.input}
        
        Here are the search results to use:
        {context}

        Here are the source URLs to cite in [#] format:{url_text}

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
            7. Do not include any additional information or references outside of the provided search results
        Please generate a comprehensive response based on the above information."""

        prompt_messages = [
            ("system", self.system_prompt),
            ("human", instruction),
            ("placeholder", "{conversation}")
        ]

        # Create template and invoke with conversation
        template = ChatPromptTemplate(messages=prompt_messages)
        return template.invoke({"conversation": messages})

    async def __call__(self, state: AgentState):
        """Async generator for responding to queries"""
        print(Fore.GREEN + "\n✨ Generating final response" + Style.RESET_ALL)
        start_time = time.time()
        response = ""

        if state.plan.direct_response:
            # Split the direct response into chunks and yield each chunk
            response = state.plan.direct_response
            chunk_size = 100  # Define the chunk size
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                yield chunk
        else:
            messages = self._build_messages(state)
            # Stream the response
            async for chunk in self.model.astream(messages):
                if hasattr(chunk, 'content'):
                    if isinstance(chunk.content, list):
                        # Claude 3 format
                        for content_item in chunk.content:
                            if content_item.get('type') == 'text':
                                token = content_item.get('text', '')
                                if token:
                                    response += token
                                    yield token
                    else:
                        token = chunk.content
                        if token:
                            response += token
                            yield token
            
        # Yield final message with history
        yield {
            'response': response,
            'message_history': [*state.message_history, {
                "role": "assistant",
                "content": response
            }]
        }
        elapsed_time = time.time() - start_time
        print(Fore.YELLOW + f"\n⏱️ Response generation took: {elapsed_time:.2f} seconds" + Style.RESET_ALL)

    async def acall(self, state: AgentState) -> AgentState:
        """Async execution with streaming events"""
        # Generate response
        response = await self.model.ainvoke(self._build_messages(state))
        result = cast(str, response.content)
        
        # Return state with response
        return {
            **state.model_dump(),
            "response": result,
        }

    async def astream(self, state: AgentState):
        """Async streaming execution"""
        print(Fore.GREEN + "\n✨ Generating final response" + Style.RESET_ALL)
        start_time = time.time()

        # If we have a direct response, use it without streaming
        if state.plan.direct_response:
            response = state.plan.direct_response
            yield response
        else:
            messages = self._build_messages(state)
            # Stream the response
            complete_response = ""
            async for chunk in self.model.astream(messages):
                if hasattr(chunk, 'content'):
                    if isinstance(chunk.content, list):
                        # Claude 3 format
                        for content_item in chunk.content:
                            if content_item.get('type') == 'text':
                                token = content_item.get('text', '')
                                if token:
                                    complete_response += token
                                    yield token
                    else:
                        token = chunk.content
                        if token:
                            complete_response += token
                            yield token
            
            # Yield final message with history
            yield {
                'response': complete_response,
                'message_history': [*state.message_history, {
                    "role": "assistant",
                    "content": complete_response
                }]
            }

        elapsed_time = time.time() - start_time
        print(Fore.YELLOW + f"⏱️ Response generation took: {elapsed_time:.2f} seconds" + Style.RESET_ALL)
