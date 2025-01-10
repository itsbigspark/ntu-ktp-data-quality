import pandas as pd
from data_validator.validation import validate_patterns

def test_validate_patterns():
    data = pd.DataFrame({"col": ["123", "abc", "789"]})
    regex_patterns = {"col": r"^\d+$"}
    invalid = validate_patterns(data, regex_patterns)
    assert len(invalid) == 1
