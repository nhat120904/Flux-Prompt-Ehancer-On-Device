from prompt_enhancer import is_complete_sentence


def test_sentence_completion() -> None:
    assert is_complete_sentence("hello.")
    assert is_complete_sentence("hello!")
    assert is_complete_sentence("hello?")
    assert not is_complete_sentence("hello")
