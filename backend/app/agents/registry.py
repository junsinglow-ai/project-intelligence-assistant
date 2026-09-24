"""Agent registry with automatic discovery.

Adding an agent = create a module in app/agents/ with a BaseAgent subclass
decorated by @register_agent. No changes to the router or API are needed.
"""

import importlib
import pkgutil

from langchain_core.tools import BaseTool

from app.agents.base import BaseAgent

_REGISTRY: dict[str, type[BaseAgent]] = {}

# Modules in this package that are infrastructure, not agents.
_NON_AGENT_MODULES = {"base", "registry"}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    if not getattr(cls, "name", None) or not getattr(cls, "description", None):
        raise ValueError(f"{cls.__name__} must define 'name' and 'description'")
    # A skill that is a bare function rather than a tool fails deep inside the
    # model call, by which point the traceback names neither the agent nor the
    # skill. Registration is where the agent is still identifiable.
    for skill in getattr(cls, "skills", []):
        if not isinstance(skill, BaseTool):
            raise ValueError(
                f"{cls.__name__} skill {skill!r} is not a tool; decorate it with @tool"
            )
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Duplicate agent name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def discover_agents() -> None:
    """Import every module in app.agents so their decorators run."""
    import app.agents as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name not in _NON_AGENT_MODULES:
            importlib.import_module(f"{pkg.__name__}.{mod.name}")


def get_agent(name: str) -> BaseAgent:
    discover_agents()
    return _REGISTRY[name]()


def list_agents(include_router: bool = False) -> list[type[BaseAgent]]:
    discover_agents()
    return [a for n, a in _REGISTRY.items() if include_router or n != "router"]
