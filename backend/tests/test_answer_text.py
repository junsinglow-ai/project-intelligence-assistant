"""The answer is the model's text, not a repr of its content blocks.

Models differ in what `AIMessage.content` holds: a plain string on some, a list
of typed blocks on others (Gemini 3.x returns text blocks carrying a reasoning
signature). Stringifying the list puts `[{'type': 'text', ...}]` in front of the
user, and it has to be caught here because the scripted model in the unit tests
only ever produces the plain-string form.
"""

from langchain_core.messages import AIMessage

from app.agents.base import last_message


def test_a_plain_string_answer_passes_through():
    state = {"messages": [AIMessage(content="The budget is 6,315,000 USD [1].")]}

    assert last_message(state) == "The budget is 6,315,000 USD [1]."


def test_structured_content_blocks_are_flattened_to_their_text():
    state = {"messages": [AIMessage(content=[
        {"type": "text", "text": "The budget is 6,315,000 USD [1]."},
        {"type": "text", "text": " Spend is on plan [2]."},
    ])]}

    answer = last_message(state)

    assert answer == "The budget is 6,315,000 USD [1]. Spend is on plan [2]."
    assert "'type'" not in answer


def test_no_messages_is_an_empty_answer_rather_than_an_error():
    assert last_message({}) == ""
