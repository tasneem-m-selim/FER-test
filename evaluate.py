import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import os

# Arguments
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <submission_csv> <labels_csv>")
    sys.exit(1)

submission_csv = sys.argv[1]   # e.g., submission/team3.csv
labels_csv = sys.argv[2]       # e.g., private_data/test_labels.csv

# Extract team name from CSV filename
team_name = os.path.splitext(os.path.basename(submission_csv))[0]

# Load CSVs
submission = pd.read_csv(submission_csv)
labels = pd.read_csv(labels_csv)

# Merge and compute accuracy
merged = pd.merge(labels, submission, on="id")
acc = accuracy_score(merged["emotion"], merged["predicted_label"])
print(f"Accuracy for {team_name}: {acc*100:.2f}%")

# Save evaluated submission to results folder
os.makedirs("results", exist_ok=True)
result_file = f"results/{team_name}_acc_{acc*100:.2f}.csv"
merged.to_csv(result_file, index=False)
print(f"Saved evaluated submission to {result_file}")
