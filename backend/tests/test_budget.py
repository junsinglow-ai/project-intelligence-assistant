"""The tool budget: a spent budget withdraws the skills rather than just asking.

Gemma 4 26B was measured ignoring `BUDGET_SPENT` and asking for more tools until
the recursion limit discarded its answer, so a message alone is not enough.
"""

import asyncio
from dataclasses import dataclass, field, replace

from app.skills.evidence import EnforceBudget, current_evidence, evidence_scope


@dataclass
class FakeRequest:
    tools: list = field(default_factory=lambda: ["search_documents", "run_sql"])

    def override(self, **overrides):
        return replace(self, **overrides)


def _offered(budget, calls_made):
    """The tools the model is offered after `calls_made` of `budget` are spent."""
    seen = []

    async def handler(request):
        seen.append(request.tools)

    async def main():
        with evidence_scope(budget=budget):
            for _ in range(calls_made):
                current_evidence().start_call()
            await EnforceBudget().awrap_model_call(FakeRequest(), handler)

    asyncio.run(main())
    return seen[0]


def test_skills_stay_offered_while_budget_remains():
    assert _offered(budget=2, calls_made=1) == ["search_documents", "run_sql"]


def test_a_spent_budget_withdraws_every_skill():
    assert _offered(budget=2, calls_made=2) == []


def test_a_zero_budget_is_unbounded_not_spent():
    assert _offered(budget=0, calls_made=5) == ["search_documents", "run_sql"]


def test_outside_an_agent_run_the_request_is_untouched():
    seen = []

    async def handler(request):
        seen.append(request.tools)

    asyncio.run(EnforceBudget().awrap_model_call(FakeRequest(), handler))
    assert seen == [["search_documents", "run_sql"]]
