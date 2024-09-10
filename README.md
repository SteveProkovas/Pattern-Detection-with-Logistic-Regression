# Pattern Detection with Logistic Regression

This project demonstrates a simple machine learning algorithm in Python designed to detect the pattern `1234` within sequences of numbers. The model is implemented using Logistic Regression and addresses the challenges of class imbalance in the dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to detect the occurrence of the sequence `[1, 2, 3, 4]` within randomly generated sequences of numbers. The project includes:

- **Data Generation**: Creation of synthetic data with and without the target pattern.
- **Feature Extraction**: Transforming sequences into feature vectors.
- **Model Training**: Using Logistic Regression with class weighting to handle data imbalance.
- **Evaluation**: Measuring model performance with appropriate metrics such as precision, recall, and F1 score.

## Getting Started

### Prerequisites

Make sure you have Python installed along with the necessary libraries:

```bash
pip install numpy scikit-learn
```

### Cloning the Repository

```bash
git clone https://github.com/SteveProkovas/pattern-detection-logistic-regression.git
cd pattern-detection-logistic-regression
```

## Usage

To run the code, simply execute the Python script:

```bash
python pattern_detection.py
```

This will generate synthetic data, train the model, and output evaluation metrics.

## Evaluation

The model is evaluated using accuracy, precision, recall, and F1 score. Due to the class imbalance in the dataset, the `class_weight='balanced'` parameter is used in the Logistic Regression model to improve performance on the minority class.

Example output:

```plaintext
Accuracy: 0.95
              precision    recall  f1-score   support

           0       0.96      0.98      0.97       190
           1       0.88      0.80      0.84        10

    accuracy                           0.95       200
   macro avg       0.92      0.89      0.90       200
weighted avg       0.95      0.95      0.95       200
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
