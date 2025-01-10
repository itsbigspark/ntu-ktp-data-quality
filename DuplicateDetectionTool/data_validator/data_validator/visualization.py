import matplotlib.pyplot as plt

def visualize_metrics(history):
    """Visualize impurity and FPR over iterations."""
    impurity = [h["impurity"] for h in history]
    fpr = [h["fpr"] for h in history]

    plt.plot(impurity, label="Impurity", marker="o")
    plt.plot(fpr, label="FPR", marker="o")
    plt.xlabel("Iterations")
    plt.ylabel("Metrics")
    plt.legend()
    plt.show()
