# At the top of the file, import get_mcp_config instead of MCP_SERVER_CONFIG
from agent.agents import create_agents, get_mcp_config

import sys
from pathlib import Path
import os

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import streamlit as st

# Inject secrets into environment BEFORE importing agent modules
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import subprocess
import os


# Now import agent modules — they'll see the correct environment
import asyncio
from dotenv import load_dotenv
from agents import Runner
from agents.mcp import MCPServerStdio
from agent.agents import create_agents, get_mcp_config

load_dotenv()

# Import after setting environment variables
from agents import Runner
from agents.mcp import MCPServerStdio
from agent.agents import create_agents, MCP_SERVER_CONFIG

st.set_page_config(page_title="Research Agent", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@600;700&family=Inter:wght@400;500&display=swap');
html, body {
    font-family: serif !important;
}

.stMarkdown, .stText, p, li, label, .stCaption {
    font-family: serif !important;
}
h1, h2, h3 { font-family: serif !important; }

/* Suggestion buttons */
div[data-testid="stButton"] > button {
    background: #171b26 !important;
    border: 1px solid #2a2f45 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 14px 16px !important;
    height: auto !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.4 !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #4ade9e !important;
    color: #4ade9e !important;
    background: rgba(74,222,158,0.05) !important;
}

/* Chat input box */
div[data-testid="stChatInput"] textarea {
    background: #171b26 !important;
    border: 2px solid #4ade9e !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
    padding: 14px !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #4ade9e !important;
    box-shadow: 0 0 0 3px rgba(74,222,158,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Research Agent")
st.caption("Backed by a multi-agent pipeline — Orchestrator → DataAgent → AnalysisAgent → InterpretationAgent")

st.divider()

# Suggestion buttons
st.markdown("**Try a question:**")

suggestions = [
    ("📈 Volatility", "What is the current GARCH volatility regime and 5-day forecast?"),
    ("📅 Seasonality", "Is there evidence of an annual seasonal pattern in Glencore returns?"),
    ("🔗 Commodities", "Does copper Granger-cause Glencore? Are they cointegrated?"),
    ("📋 Research Note", "Give me a full research note on Glencore's current state across all analyses."),
    ("⚡ Signals", "What are the current technical signals? RSI, momentum, vol ratio."),
]

cols = st.columns(len(suggestions))
for col, (icon_label, full_q) in zip(cols, suggestions):
    if col.button(icon_label, key=f"sug_{icon_label}", use_container_width=True):
        st.session_state["prefill"] = full_q

st.divider()

# Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prefill  = st.session_state.pop("prefill", "")
question = st.chat_input("Ask a research question about Glencore...")
if not question and prefill:
    question = prefill

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running analysis pipeline..."):
            async def run_agent(q):
                async with MCPServerStdio(
                     params=get_mcp_config(),
                    cache_tools_list=True,
                    client_session_timeout_seconds=60.0,
                ) as mcp:
                    orchestrator = create_agents(mcp)
                    result = await Runner.run(orchestrator, input=q, max_turns=20)
                    return result.final_output

            try:
                answer = asyncio.run(run_agent(question))
            except Exception as e:
                st.exception(e)
                answer = f"Agent error: {repr(e)}"

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

if st.session_state["messages"]:
    if st.button("Clear chat", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()