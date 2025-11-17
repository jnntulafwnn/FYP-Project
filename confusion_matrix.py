import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Define your categories
categories = ["Cooking Essentials", "Rice & Grains", "Household", "Bakery", "Canned & Packaged"]

# Ground truth (actual categories)
y_true = [
    "Canned & Packaged", "Rice & Grains", "Household", "Bakery",
    "Canned & Packaged", "Rice & Grains", "Household", "Cooking Essentials", "Bakery",
    "Rice & Grains", "Cooking Essentials", "Cooking Essentials", "Cooking Essentials"
]

# Predicted categories (all correct)
y_pred = y_true.copy()

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=categories)

# Plot the matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Category', fontsize=12)
plt.ylabel('Actual Category', fontsize=12)
plt.title('Confusion Matrix for Expense Categorization Model', fontsize=14)
plt.tight_layout()
plt.show()
