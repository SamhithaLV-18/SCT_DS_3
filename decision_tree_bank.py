# ------------------------------------------------------
# SIMPLE & CLEAR DECISION TREE (NO GRAPHVIZ)
# Matplotlib only – No overlapping
# ------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

print("Loading dataset...")
df = pd.read_csv("bank-full.csv", sep=';')
print("Dataset Loaded!\n")

# -------------------------------
# ENCODE CATEGORICAL DATA
# -------------------------------

data = df.copy()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop("y", axis=1)
y = data["y"]

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN DECISION TREE
# -------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,   # 🔥 SMALL TREE = NO OVERLAP
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# SIMPLE VISUALIZATION
# -------------------------------

plt.figure(figsize=(20, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.tight_layout()
plt.show()

print("\nDecision tree displayed successfully (simple version, no overlap).")
