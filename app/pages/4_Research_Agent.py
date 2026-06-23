import sys
from pathlib import Path

# Add project root so both 'src' and 'agent' packages are found
_root = Path(__file__).parent.parent.parent  # app/pages/ -> app/ -> project root
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import asyncio
import streamlit as st
from dotenv import load_dotenv
from agents import Runner
from agents.mcp import MCPServerStdio
from agent.agents import create_agents, MCP_SERVER_CONFIG

load_dotenv()

st.set_page_config(page_title="Research Agent", layout="wide")
st.title("🤖 Research Agent")
st.caption("Ask any research question about Glencore — the agent runs the full analysis pipeline.")

# Suggested questions
st.subheader("Suggested questions")
suggestions = [
    "What is the current volatility regime? Give me the GARCH forecast.",
    "Is there evidence of seasonal patterns in Glencore returns?",
    "Does copper Granger-cause Glencore? Are they cointegrated?",
    "Give me a full research note on Glencore's current state.",
    "What are the current technical signals? Is the stock overbought?",
]
cols = st.columns(len(suggestions))
for i, (col, q) in enumerate(zip(cols, suggestions)):
    if col.button(q[:40] + "...", key=f"sug_{i}"):
        st.session_state["prefill"] = q

st.divider()

# Chat history — persists across reruns in the session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask a research question about Glencore...")
if not question and prefill:
    question = prefill

if question:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Research agent running..."):
            async def run_agent(q):
                async with MCPServerStdio(
                    params=MCP_SERVER_CONFIG,
                    cache_tools_list=True,
                    client_session_timeout_seconds=60.0,
                ) as mcp:
                    orchestrator = create_agents(mcp)
                    result = await Runner.run(orchestrator, input=q, max_turns=20)
                    return result.final_output

            try:
                answer = asyncio.run(run_agent(question))
            except Exception as e:
                answer = f"Agent error: {e}"

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

if st.button("Clear chat", type="secondary"):
    st.session_state["messages"] = []
    st.rerun()