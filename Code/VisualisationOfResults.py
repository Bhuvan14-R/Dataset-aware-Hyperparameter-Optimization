import json
import matplotlib.pyplot as plt

# Load results
with open(r"D:\Project\ML\Results\Output\results.json", "r") as f:
    data = json.load(f)

iris_acc = data["iris"]["accuracy"]
digits_acc = data["digits"]["accuracy"]
wine_acc = data["wineqt"]["accuracy"]
meta_acc = data["meta_learning"]["accuracy"]

# -------------------------------
# GRAPH 1 — Accuracy
# -------------------------------
plt.figure()

methods = ["Iris", "Digits", "WineQT", "Meta"]
accuracies = [iris_acc, digits_acc, wine_acc, meta_acc]

bars = plt.bar(methods, accuracies)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2f}", ha='center')

plt.tight_layout()
plt.savefig(r"D:\Project\ML\Results\Graph\accuracy_comparison.png", dpi=300)
plt.close()


# -------------------------------
# GRAPH 2 — Trials
# -------------------------------
plt.figure()

methods = ["Optuna", "Meta"]
trials = [20, 1]

bars = plt.bar(methods, trials)
plt.title("Trials Comparison")
plt.ylabel("Number of Trials")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, str(int(yval)), ha='center')

plt.tight_layout()
plt.savefig(r"D:\Project\ML\Results\Graph\trials_comparison.png", dpi=300)
plt.close()


# -------------------------------
# GRAPH 3 — n_estimators
# -------------------------------
plt.figure()

labels = ["Optuna", "Predicted"]
values = [
    data["iris"]["best_params"]["n_estimators"],
    data["meta_learning"]["n_estimators"]
]

bars = plt.bar(labels, values)
plt.title("n_estimators Comparison")
plt.ylabel("Value")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(int(yval)), ha='center')

plt.tight_layout()
plt.savefig(r"D:\Project\ML\Results\Graph\n_estimators.png", dpi=300)
plt.close()


# -------------------------------
# GRAPH 4 — max_depth
# -------------------------------
plt.figure()

labels = ["Optuna", "Predicted"]
values = [
    data["iris"]["best_params"]["max_depth"],
    data["meta_learning"]["max_depth"]
]

bars = plt.bar(labels, values)
plt.title("max_depth Comparison")
plt.ylabel("Value")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, str(int(yval)), ha='center')

plt.tight_layout()
plt.savefig(r"D:\Project\ML\Results\Graph\max_depth.png", dpi=300)
plt.close()

print("All 4 graphs generated successfully.")