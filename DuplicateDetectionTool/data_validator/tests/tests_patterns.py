import pandas as pd
from data_validator.patterns import infer_regex_patterns

def test_infer_regex_patterns():
    data = pd.DataFrame({"col": ["123", "456", "789"]})
    patterns = infer_regex_patterns(data)
    assert patterns["col"] == r"^\d+$"
