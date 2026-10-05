from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")


def test_app_loads_without_error():
    at = AppTest.from_file(APP).run()
    assert not at.exception


def test_pasting_a_passage_shows_a_reading():
    at = AppTest.from_file(APP).run()
    at.text_area[0].set_value(
        "This could damage our reputation and might not resonate with guests."
    ).run()
    assert not at.exception
    readings = {m.label: m.value for m in at.metric}
    assert readings["Reading"] == "Depreciator"


def test_example_button_fills_the_box():
    at = AppTest.from_file(APP).run()
    at.button[0].click().run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Reading"] == "Celebrator"
