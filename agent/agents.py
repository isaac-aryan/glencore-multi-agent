"""
agents.py — All agent definitions for the Glencore Research Agent system.

Architecture:
    Orchestrator → handoff → DataAgent | AnalysisAgent | InterpretationAgent
    DataAgent and AnalysisAgent use the glencore MCP server for tools.
    InterpretationAgent synthesises text only — no tools needed.

Usage:
    from agent.agents import create_agents, MCP_SERVER_CONFIG
    async with MCPServerStdio(params=MCP_SERVER_CONFIG) as mcp_server:
        orchestrator = create_agents(mcp_server)
"""

import os
from dotenv import load_dotenv
from agents import Agent
from agents.mcp import MCPServerStdio

load_dotenv()

# ── MCP server config ──────────────────────────────────────────────────────
# Tells the SDK how to spawn your MCP server as a subprocess.
# Uses stdio transport — the agent SDK and server talk over stdin/stdout.
import os

import subprocess

import sys

def get_mcp_config():
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src_path = os.path.join(_project_root, "src")
    _mcp_path = os.path.join(_src_path, "glencore_multi_agent", "mcp_server.py")
    _python = sys.executable

    return {
        "command": _python,
        "args": [_mcp_path],
        "cwd": _project_root,   # ← subprocess starts from project root
        "env": {
            **os.environ,
            "PYTHONPATH": _src_path,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        },
    }

# Keep MCP_SERVER_CONFIG as a property for backwards compatibility
MCP_SERVER_CONFIG = get_mcp_config()

# ── Agent factory ──────────────────────────────────────────────────────────

def create_agents(mcp_server: MCPServerStdio) -> Agent:
    """
    Create all agents sharing a single connected MCP server instance.
    The server must already be connected (via async with) before calling this.

    Returns the Orchestrator, which holds references to all specialists
    via its handoffs list.
    """

    # ── Specialist: Data Agent ─────────────────────────────────────────────
    data_agent = Agent(
        name="DataAgent",
        model="gpt-4o-mini",
        instructions="""You are a financial data specialist for Glencore (GLEN.L).

Your role: retrieve and summarise market data using your available tools.

Guidelines:
- Always call get_glencore_data() before answering any price or return question
- Call get_dividend_history() when the question involves dividends or the annual cycle hypothesis
- Call get_rolling_statistics() for current volatility regime questions
- Return structured data clearly — include dates, units, and data quality notes
- Never extrapolate or predict — only report what the data shows
- Always state units: prices are in GBX (pence), not GBP (pounds)

Example: a price of 450 means 450 pence = £4.50.""",
        mcp_servers=[mcp_server],
    )

    # ── Specialist: Analysis Agent ─────────────────────────────────────────
    analysis_agent = Agent(
        name="AnalysisAgent",
        model="gpt-4o",
        instructions="""You are a quantitative analyst specialising in time series and financial econometrics.

Your role: run statistical tests using your tools and interpret the results rigorously.

When analysing seasonality:
- Call run_seasonality_analysis() to get ACF values, monthly returns, and the F-test
- Note that with ~15 years of annual data, statistical power is limited
- The dividend hypothesis is always the first alternative explanation to consider
- Distinguish clearly between 'statistically significant' and 'practically meaningful'

When analysing volatility:
- Call fit_garch() to fit a GARCH(1,1) model
- Interpret the persistence parameter (alpha + beta): near 1.0 means shocks decay slowly
- Report the 5-day volatility forecast in the context of the current vol regime

Always:
- State what test was run and what its null hypothesis is
- Report both the test statistic and the p-value
- Give a plain-English conclusion alongside the technical result
- Be honest about limitations: sample size, structural breaks, data gaps""",
        mcp_servers=[mcp_server],
    )

    # ── Specialist: Interpretation Agent ──────────────────────────────────
    interpretation_agent = Agent(
        name="InterpretationAgent",
        model="gpt-4o",
        instructions="""You are a research communicator translating quantitative findings into clear insights.

Your role: take statistical results and data summaries from other agents and write a
concise, honest research note.

Always structure your output as:
1. Finding — one sentence stating what was found (or not found)
2. Evidence — what the data and tests showed, with numbers and units
3. Caveats — alternative explanations (e.g. dividends, small sample, regime breaks)
4. Confidence — how much weight to place on this finding given sample size and test power

Intellectual honesty rules:
- A negative result (no pattern found) is a valid and complete finding — say so clearly
- Never overstate statistical significance
- If tests are inconclusive, say so rather than picking a side
- Always note that this is a research exercise, not a trading signal""",
        # No mcp_servers — this agent synthesises text only, no tools needed
    )

    # ── Orchestrator ───────────────────────────────────────────────────────
    orchestrator = Agent(
        name="OrchestratorAgent",
        model="gpt-4o",
        instructions="""You are the orchestrator of a Glencore quantitative research system.
You receive research questions and coordinate specialist agents to answer them rigorously.

Your specialists and when to use them:
- DataAgent: for retrieving price data, dividends, rolling statistics
- AnalysisAgent: for running statistical tests (stationarity, seasonality, GARCH)
- InterpretationAgent: for synthesising findings into a clear research note

Workflow for research questions (e.g. "is there a seasonal pattern?"):
1. Hand off to DataAgent to retrieve relevant data and context
2. Hand off to AnalysisAgent to run the appropriate statistical tests
3. Pass all results to InterpretationAgent for the final write-up
4. Return the InterpretationAgent's synthesis as your final answer

Workflow for simple factual questions (e.g. "what is the current price?"):
1. Hand off to DataAgent only — no analysis or interpretation needed

You MUST always hand off to AnalysisAgent to run statistical tests before 
handing off to InterpretationAgent. Never return a final answer without 
running at least one statistical test.

Always be honest about what the analysis can and cannot conclude.
Never fabricate statistical results — always use tool outputs.""",
        handoffs=[data_agent, analysis_agent, interpretation_agent],
    )

    return orchestrator