import pandas as pd
from data_validator.metrics import calculate_metrics

def test_calculate_metrics():
    data = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    metrics = calculate_metrics(data)
    assert "mean" in metrics["col"]
