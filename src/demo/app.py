import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import streamlit as st
from src.agent.states.schemas import AgentState
from src.agent.workflow import SearchAgentWorkflow
from src.agent.llm.responder_model import ResponderModel
from src.agent.nodes.responder import Responder
from src.monitoring.phoenix import setup_phoenix

# ===== Constants =====
PAGE_CONFIG = {
    "page_title": "AI Research Assistant",
    "page_icon": "✨",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

COLUMN_CONFIG = {
    "seq": st.column_config.NumberColumn(
        "№",
        help="Sequence number",
        format="%d",
        width="small",
    ),
    "title": st.column_config.TextColumn(
        "Title",
        help="Article title",
        max_chars=100,
        width="large",
    ),
    "url": st.column_config.LinkColumn(
        "Source",
        display_text="View Source",
        help="Click to open source",
        width="medium",
    ),
    "score": st.column_config.NumberColumn(
        "Relevance",
        help="Search relevance score",
        format="%.2f",
        width="small",
    ),
}

# ===== Set Streamlit page config =====
st.set_page_config(**PAGE_CONFIG)

# ===== Initialize Streamlit session state =====
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState.initial_state()
if "tracer_provider" not in st.session_state:
    st.session_state.tracer_provider = setup_phoenix()
if "workflow" not in st.session_state:
    st.session_state.workflow = SearchAgentWorkflow(
        tracer_provider=st.session_state.tracer_provider
    )
if "responder" not in st.session_state:
    responder_model = ResponderModel(st.session_state.tracer_provider)
    st.session_state.responder = Responder(responder_model)

# ===== Set Streamlit layout =====
st.title("✨ AI Research Assistant")

# Display chat history
for message in st.session_state.agent_state.message_history:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "✨"):
        st.write(message["content"])

# ===== Workflow Functions =====
def process_search_results(task_info, executed_info, executed_tasks_info):
    """Process and display search results in a table format"""
    if isinstance(task_info.get('result'), dict) and 'results' in task_info['result']:
        search_results = task_info['result']['results']
        if search_results:
            df = pd.DataFrame(search_results)
            df.insert(0, 'seq', range(1, len(df) + 1))
            
            # Search results in data editor
            executed_tasks_info.data_editor(
                df[["seq", "title", "url", "score"]],
                column_config=COLUMN_CONFIG,
                hide_index=True,
                num_rows="dynamic",
            )
            executed_info.markdown("---")
    else:
        with executed_info.expander(task_info["task"].title, expanded=True):
            st.info(task_info.get('result', 'No results available'))

async def run_workflow(state: AgentState):
    final_state = AgentState(**state.model_dump())
    streaming_response = False
    
    async for event in st.session_state.workflow.astream(state):
        if "planner" in event:
            if "in_progress" in event.get("status", ""):
                token = event.get("planner", {}).get("token")
                if token:
                    yield {"type": "plan_token", "content": token}
            else:
                final_state.plan = event["planner"]["plan"]
                yield {"type": "plan_complete", "content": event["planner"]["plan"]}
        
        elif "executor" in event:
            final_state.executed_tasks = event["executor"]["executed_tasks"]
            final_state.remaining_tasks = event["executor"]["remaining_tasks"]
            
            if "executed_tasks" in event["executor"]:
                yield {"type": "execute_progress", "content": event["executor"]["executed_tasks"]}
            if "remaining_tasks" in event["executor"]:
                yield {"type": "execute_remaining", "content": event["executor"]["remaining_tasks"]}
        
        elif "responder" in event:
            if "in_progress" in event.get("status", ""):
                token = event.get("responder", {}).get("token")
                if token:
                    streaming_response = True
                    yield {"type": "response_token", "content": token}
            else:
                if not streaming_response and "response" in event.get("responder", {}):
                    yield {"type": "response_token", "content": event["responder"]["response"]}
    
    yield {"type": "final", "content": final_state}

async def process_workflow():
    progress_status.update(label="🤔 Starting research...", state="running") 
    response = ""
    
    async for event in run_workflow(st.session_state.agent_state):
        try:
            if event["type"] == "plan_token":
                progress_status.update(label="🎯 Planning research...", state="running", expanded=False)
                
            elif event["type"] == "plan_complete":
                progress_status.update(label="✅ Research plan ready!", state="complete", expanded=True)
                plan = event["content"]
                if plan.requires_tool:
                    with plan_info:
                        st.write(event["content"])
                    progress_status.update(label="🔍 Gathering information...", state="running")
                    
            elif event["type"] == "execute_progress":
                executed_tasks = event["content"]
                for task_info in executed_tasks:
                    process_search_results(task_info, executed_info, executed_tasks_info)
                
            elif event["type"] == "execute_remaining":
                remaining = event["content"]
                if not remaining:
                    progress_status.update(label="✨ Analysis complete!", state="complete", expanded=False)
                else:
                    executed_info.info("Remaining tasks: " + ", ".join([str(task) for task in remaining]))
                    
            elif event["type"] == "response_token":
                progress_status.update(label="✨ Providing insights!", state="complete", expanded=False)
                response += event["content"]
                yield event["content"]

        except Exception as e:
            st.warning(f"Warning during processing: {str(e)}")
            continue

    st.session_state.agent_state.message_history.append({
        "role": "assistant",
        "content": response
    })

# ===== Main Chat Interface =====
if prompt := st.chat_input("💭 What would you like to explore?"):
    st.chat_message("user", avatar="👤").write(prompt)
    st.session_state.agent_state.input = prompt
    st.session_state.agent_state.message_history.append({
        "role": "user",
        "content": prompt
    })
    
    try:
        # Create status containers
        with st.status(label="🤔 Starting research...", expanded=True) as progress_status:
            # Create two columns for better layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                plan_info = st.empty()
                executed_info = st.empty()
            with col2:
                executed_tasks_info = st.empty()
       
        answer_container = st.chat_message("assistant", avatar="✨")
        answer_container.write_stream(process_workflow())
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")