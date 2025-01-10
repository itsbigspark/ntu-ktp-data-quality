from data_validator.synthesis import generate_synthetic_data

def test_synthetic_data():
    metrics = {"col": {"mean": 5, "std": 2}}
    data = generate_synthetic_data(metrics, {}, 10)
    assert len(data) == 10
