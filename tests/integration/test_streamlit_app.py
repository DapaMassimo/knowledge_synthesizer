import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

_APP = "src/knowledge_synthesizer/entrypoints/streamlit_app.py"


def test_app_renders_without_interaction() -> None:
    app = AppTest.from_file(_APP).run(timeout=60)

    assert not app.exception
    assert app.title[0].value == "Knowledge Synthesizer"
    # Inputs are present and no service was invoked (no network) on first render.
    assert any(button.label == "Index" for button in app.button)
