"""
runner.py — Entry point for the Glencore Research Agent.

The MCP server is connected once via async context manager and shared
across all agents for the duration of the run.

Usage:
    python -m agent.runner "Is there a seasonal pattern in Glencore?"
    python -m agent.runner "What is the current volatility regime?"
    python -m agent.runner "Fit a GARCH model and interpret the persistence"
    python -m agent.runner  # runs default question if no argument given
"""

import asyncio
import sys
from dotenv import load_dotenv
from agents import Runner
from agents.mcp import MCPServerStdio
from agent.agents import create_agents, MCP_SERVER_CONFIG

load_dotenv()

DEFAULT_QUESTION = (
    "Give me a quick data summary for Glencore: "
    "current price, 1-year return, and annualised volatility."
)


async def run_query(question: str) -> str:
    """
    Run a research question through the full agent pipeline.

    Opens an MCP server connection, builds the agent system,
    runs the query, then cleans up the server on exit.
    """
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")

    # Connect the MCP server once — shared by all agents in this run.
    # async with handles connect() on entry and cleanup() on exit automatically.
    async with MCPServerStdio(
    params=MCP_SERVER_CONFIG,
    cache_tools_list=True,
    client_session_timeout_seconds=60.0,
    ) as mcp_server:
        orchestrator = create_agents(mcp_server)

        result = await Runner.run(
            orchestrator,
            input=question,
            max_turns=20,  # safety limit — prevents infinite loops during development
        )

    print("\n── FINAL ANSWER ──────────────────────────────────────────")
    print(result.final_output)
    return result.final_output


async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION
    await run_query(question)


if __name__ == "__main__":
    asyncio.run(main())