import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to generate synthetic data
def generate_data(num_sequences, sequence_length):
    X = []
    y = []
    pattern = [1, 2, 3, 4]

    for _ in range(num_sequences):
        if np.random.rand() > 0.25:  # 25% chance to include the pattern
            start_index = np.random.randint(0, sequence_length - 4)
            sequence = np.random.randint(1, 10, sequence_length)
            sequence[start_index:start_index + 4] = pattern
            y.append(1)  # Pattern present
        else:
            sequence = np.random.randint(1, 10, sequence_length)
            y.append(0)  # Pattern absent

        X.append(sequence)

    return np.array(X), np.array(y)

# Function to extract features from sequences
def extract_features(sequences):
    features = []
    for seq in sequences:
        features.append([
            np.sum(seq == 1),
            np.sum(seq == 2),
            np.sum(seq == 3),
            np.sum(seq == 4),
            len(seq),
            np.mean(seq),               # New feature: Mean of the sequence
            np.std(seq),                # New feature: Standard deviation
            np.max(seq),                # New feature: Maximum value
            np.min(seq),                # New feature: Minimum value
        ])
    return np.array(features)

# Generate synthetic dataset with increased size
X, y = generate_data(num_sequences=5000, sequence_length=10)  # Increased dataset size
X_features = extract_features(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV for Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

model = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5)  # Using cross-validation with 5 folds
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
