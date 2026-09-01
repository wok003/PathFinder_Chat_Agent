import pytest 
from orchestrator.agent_builder import Toolbox

@pytest.fixture
def toolbox_init():
    return Toolbox(max_results=5)

@pytest.mark.parametrize("text_input, expected_output", [
    ("abcdefghijklmnopqrstuvwxyz",  "zyxwvutsrqponmlkjihgfedcba"),
    ("12345", "54321"),
    ("",""),
    ("a","a")
])
def test_string_reversal(text_input, expected_output, toolbox_init):
    reversed_text = toolbox_init.reverse_string(text_input)
    assert reversed_text == expected_output
