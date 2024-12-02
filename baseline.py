import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
data = pd.read_csv("tweets.csv")

# Step 1: Define the distress keywords
distress_keywords = ["trapped", "emergency", "sos", "need help", "injured", "rescue", "lost", "missing"]

# Step 2: Define the baseline models
def keyword_baseline(tweet):
    return int(any(keyword in tweet for keyword in distress_keywords))

def majority_class_baseline(data):
    # Determine the majority class
    majority_class = data["Label"].mode()[0]
    return [majority_class] * len(data)

# Step 3: Apply baseline models
data['Keyword_Baseline_Pred'] = data['Clean Tweet Text'].apply(keyword_baseline)
majority_baseline_preds = majority_class_baseline(data)

# Step 4: Evaluate the baselines
# Keyword-based baseline metrics
keyword_precision = precision_score(data['Label'], data['Keyword_Baseline_Pred'], zero_division=1)
keyword_recall = recall_score(data['Label'], data['Keyword_Baseline_Pred'], zero_division=1)
keyword_accuracy = accuracy_score(data['Label'], data['Keyword_Baseline_Pred'])
keyword_f1 = f1_score(data['Label'], data['Keyword_Baseline_Pred'], zero_division=1)
keyword_conf_matrix = confusion_matrix(data['Label'], data['Keyword_Baseline_Pred'])
keyword_tn, keyword_fp, keyword_fn, keyword_tp = keyword_conf_matrix.ravel()

# Majority-class baseline metrics
majority_precision = precision_score(data['Label'], majority_baseline_preds, zero_division=1)
majority_recall = recall_score(data['Label'], majority_baseline_preds, zero_division=1)
majority_accuracy = accuracy_score(data['Label'], majority_baseline_preds)
majority_f1 = f1_score(data['Label'], majority_baseline_preds, zero_division=1)
majority_conf_matrix = confusion_matrix(data['Label'], majority_baseline_preds)
majority_tn, majority_fp, majority_fn, majority_tp = majority_conf_matrix.ravel()

# Print results
print("Keyword-Based Baseline Metrics:")
print("Precision:", keyword_precision)
print("Recall:", keyword_recall)
print("Accuracy:", keyword_accuracy)
print("F1 Score:", keyword_f1)
print("Confusion Matrix:")
print("True Negatives:", keyword_tn)
print("False Positives:", keyword_fp)
print("False Negatives:", keyword_fn)
print("True Positives:", keyword_tp)

print("\nMajority-Class Baseline Metrics:")
print("Precision:", majority_precision)
print("Recall:", majority_recall)
print("Accuracy:", majority_accuracy)
print("F1 Score:", majority_f1)
print("Confusion Matrix:")
print("True Negatives:", majority_tn)
print("False Positives:", majority_fp)
print("False Negatives:", majority_fn)
print("True Positives:", majority_tp)
