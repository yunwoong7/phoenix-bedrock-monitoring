import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
from datetime import datetime
from src.agent.states.schemas import AgentState, Plan
from src.agent.workflow import SearchAgentWorkflow
from src.monitoring.phoenix import setup_phoenix

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session states
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState.initial_state()

# Initialize the agent workflow
tracer_provider = setup_phoenix()
agent = SearchAgentWorkflow(tracer_provider=tracer_provider)

# Sidebar for chat history
with st.sidebar:
    st.title("💬 Chat History")
    
    # New chat button
    if st.button("🆕 New Chat"):
        st.session_state.agent_state = AgentState.initial_state()
        st.rerun()
    
    # # Display saved chats
    # st.divider()
    # for idx, chat in enumerate(st.session_state.chat_history):
    #     if st.button(f"📝 {chat['title']}", key=f"chat_{idx}"):
    #         st.session_state.agent_state.message_history = chat['messages']
    #         st.session_state.current_chat = idx
    #         st.rerun()

# Main chat interface
st.title("🤖 AI Research Assistant")

# Chat container with messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.agent_state.message_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User input area
if prompt := st.chat_input("What would you like to know?"):
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Update agent state
    st.session_state.agent_state.input = prompt
    st.session_state.agent_state.message_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Show assistant response with loading state
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # Run agent with current state
            result = agent.workflow.invoke(st.session_state.agent_state)
            st.session_state.agent_state = AgentState(**result)
            
            # Display streaming response
            full_response = ""
            for chunk in result['response']:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            if result['plan']['requires_tool']:
                with st.expander("🔍 View Analysis Plan"):
                    st.code(str(result['plan']))
    
    # Save chat to history if it's new
    # if st.session_state.current_chat is None:
    #     chat_title = f"Chat {len(st.session_state.chat_history) + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    #     st.session_state.chat_history.append({
    #         "title": chat_title,
    #         "messages": st.session_state.agent_state.message_history
    #     })
    #     st.session_state.current_chat = len(st.session_state.chat_history) - 1
