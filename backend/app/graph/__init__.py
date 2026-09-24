"""The chat graph: one LangGraph state machine per request.

Lives outside `app/agents/` because `discover_agents()` imports every module in
that package looking for agents; a graph module there would be imported on every
registry call and found to contain none.
"""
