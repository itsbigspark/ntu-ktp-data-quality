from data_validator.visualization import visualize_metrics

def test_visualization():
    history = [{"impurity": 0.5, "fpr": 0.3}, {"impurity": 0.2, "fpr": 0.1}]
    visualize_metrics(history)
    assert True  # Ensure no errors occur during plotting
