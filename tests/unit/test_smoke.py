import knowledge_synthesizer


def test_package_exposes_version() -> None:
    assert knowledge_synthesizer.__version__ == "0.1.0"
