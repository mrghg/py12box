

from py12box import get_data
from pathlib import Path

def test_data_path():

    assert get_data("blah") == Path(__file__).parents[1] / "py12box/data/blah"