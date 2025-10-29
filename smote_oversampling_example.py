"""
SMOTE Oversampling Example for Multi-label Classification
This example demonstrates how to handle imbalanced multi-label datasets using SMOTE.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on your example
data = {
    'time_materials_video': [22551.0, 0.0, 15000.0, 8000.0, 30000.0, 0.0, 12000.0, 0.0, 25000.0, 5000.0],
    'time_materials_document': [5683.0, 36562.0, 8000.0, 0.0, 12000.0, 40000.0, 0.0, 28000.0, 6000.0, 0.0],
    'time_materials_article': [21321.0, 0.0, 10000.0, 15000.0, 8000.0, 0.0, 18000.0, 0.0, 20000.0, 12000.0],
    'labels': [
        ['Aktif', 'Visual'],
        ['Reflektif', 'Verbal'],
        ['Aktif', 'Verbal'],
        ['Reflektif', 'Visual'],
        ['Aktif', 'Visual'],
        ['Reflektif', 'Verbal'],
        ['Aktif', 'Verbal'],
        ['Reflektif', 'Visual'],
        ['Aktif', 'Visual'],
        ['Reflektif', 'Verbal']
    ]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\nDataset shape:", df.shape)

# Convert multi-label strings to binary format
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(df['labels'])

# Create feature matrix
X = df[['time_materials_video', 'time_materials_document', 'time_materials_article']]

print("\nLabel classes:", mlb.classes_)
print("Encoded labels shape:", y_encoded.shape)
print("\nSample of encoded labels:")
for i in range(min(5, len(y_encoded))):
    print(f"Row {i}: {y_encoded[i]} -> {df['labels'].iloc[i]}")

# Check class distribution before SMOTE
print("\nClass distribution before SMOTE:")
label_counts = y_encoded.sum(axis=0)
for i, label in enumerate(mlb.classes_):
    print(f"{label}: {label_counts[i]} samples")

# Split data before applying SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Apply SMOTE for multi-label classification
# For multi-label, we need to apply SMOTE to each label separately or use specialized techniques
def apply_smote_multilabel(X, y):
    """
    Apply SMOTE to multi-label data by resampling each label separately
    and combining the results.
    """
    smote = SMOTE(random_state=42)
    X_resampled_list = []
    y_resampled_list = []

    # Get unique label combinations
    unique_combinations = np.unique(y, axis=0)

    for combo in unique_combinations:
        # Find samples with this specific label combination
        mask = np.all(y == combo, axis=1)
        X_subset = X[mask]
        y_subset = y[mask]

        if len(X_subset) < 2:  # Skip if we don't have enough samples
            X_resampled_list.append(X_subset)
            y_resampled_list.append(y_subset)
            continue

        # Apply SMOTE to this subset
        try:
            # For single label combinations, use the first label as target
            if np.sum(combo) == 1:
                target_label = np.argmax(combo)
                X_res, y_res = smote.fit_resample(X_subset, y_subset[:, target_label])
                # Convert back to multi-label format
                y_res_full = np.zeros((len(y_res), len(combo)))
                y_res_full[:, target_label] = y_res
                X_resampled_list.append(X_res)
                y_resampled_list.append(y_res_full)
            else:
                # For multiple labels, keep original samples
                X_resampled_list.append(X_subset)
                y_resampled_list.append(y_subset)
        except:
            # If SMOTE fails, keep original samples
            X_resampled_list.append(X_subset)
            y_resampled_list.append(y_subset)

    # Combine all resampled data
    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.vstack(y_resampled_list)

    return X_resampled, y_resampled

# Alternative approach: Simple SMOTE application (works but may create unrealistic combinations)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Original training size: {X_train.shape[0]}")
print(f"Resampled training size: {X_train_resampled.shape[0]}")

print("\nClass distribution after SMOTE:")
label_counts_resampled = y_train_resampled.sum(axis=0)
for i, label in enumerate(mlb.classes_):
    print(f"{label}: {label_counts_resampled[i]} samples")

# Train a classifier on the resampled data
print("\nTraining Random Forest classifier...")
classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
for i, label in enumerate(mlb.classes_):
    print(f"\n{label}:")
    print(classification_report(y_test[:, i], y_pred[:, i]))

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.3f}")

# Function to visualize the results
def plot_class_distribution(y_before, y_after, label_names):
    """Plot class distribution before and after SMOTE"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before SMOTE
    counts_before = y_before.sum(axis=0)
    axes[0].bar(label_names, counts_before)
    axes[0].set_title('Class Distribution Before SMOTE')
    axes[0].set_ylabel('Number of Samples')

    # After SMOTE
    counts_after = y_after.sum(axis=0)
    axes[1].bar(label_names, counts_after)
    axes[1].set_title('Class Distribution After SMOTE')
    axes[1].set_ylabel('Number of Samples')

    plt.tight_layout()
    plt.savefig('smote_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot the distribution
plot_class_distribution(y_train, y_train_resampled, mlb.classes_)

# Example of how to use the trained model for new predictions
def predict_learning_style(new_data):
    """
    Predict learning styles for new data
    new_data: DataFrame with columns ['time_materials_video', 'time_materials_document', 'time_materials_article']
    """
    predictions = classifier.predict(new_data)
    predicted_labels = mlb.inverse_transform(predictions)
    return predicted_labels

# Example prediction
new_sample = pd.DataFrame({
    'time_materials_video': [18000.0],
    'time_materials_document': [7000.0],
    'time_materials_article': [15000.0]
})

predicted_style = predict_learning_style(new_sample)
print(f"\nPrediction for new sample {new_sample.values[0]}: {predicted_style[0]}")

# Save the model and label encoder for future use
import joblib
joblib.dump(classifier, 'smote_classifier.pkl')
joblib.dump(mlb, 'label_encoder.pkl')
print("\nModel and label encoder saved as 'smote_classifier.pkl' and 'label_encoder.pkl'")