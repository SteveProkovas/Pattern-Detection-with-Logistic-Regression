import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Function to generate synthetic data
def generate_data(num_sequences, sequence_length):
    X = []
    y = []
    pattern = [1, 2, 3, 4]

    for _ in range(num_sequences):
        if np.random.rand() > 0.25 :  # 25% chance to include the pattern
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
        ])
    return np.array(features)


# Generate synthetic dataset
X, y = generate_data(num_sequences=1000, sequence_length=10)
X_features = extract_features(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train a logistic regression model with class weighting
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
