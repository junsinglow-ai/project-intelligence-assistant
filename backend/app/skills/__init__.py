"""Skills: the tools an agent's model can call.

Skills live outside `app/agents/` on purpose. `discover_agents()` imports every
module in that package looking for agents, so a shared helper placed there has
to be listed in `_NON_AGENT_MODULES` or it is imported pointlessly on every
registry call -- and nothing fails if you forget. Keeping them here removes the
trap entirely.

Every skill is read-only and runs in-process: retrieval, a schema listing and a
guarded read-only SELECT. That is what replaces the "answering agents have no
tools" control described in ARCHITECTURE.md section 8.1 (see DECISIONS.md D-015).
"""
