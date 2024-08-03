# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, with a significant difference between the number of normal transactions and fraudulent transactions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview

Credit card fraud detection is crucial for preventing financial losses and protecting consumers. This project utilizes a logistic regression model to identify fraudulent transactions.

## Dataset

The dataset contains credit card transactions, including features transformed using PCA and a target variable indicating whether the transaction is fraudulent (1) or not (0).

## Dependencies

- numpy
- pandas
- scikit-learn

You can install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn

Data Preprocessing
Load the Dataset
python
Copy code
data = pd.read_csv('/content/creditcard.csv')
Remove Missing Values
python
Copy code
data = data.dropna()
Class Distribution
Legitimate Transactions: 0
Fraudulent Transactions: 1
Under-sampling
Build a sample dataset containing an equal distribution of normal and fraudulent transactions. Number of Fraudulent Transactions: 88

Split Data
python
Copy code
X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
Model Training
Logistic Regression
python
Copy code
model = LogisticRegression(solver='liblinear', C=0.01)
model.fit(X_train, Y_train)
Model Evaluation
Accuracy Score
python
Copy code
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data:', test_data_accuracy)
Usage
To run the project, follow these steps:

Clone the repository
bash
Copy code
git clone https://github.com/srutz1/Fraud-Detection.git
Navigate to the project directory
bash
Copy code
cd Fraud-Detection
Run the script
bash
Copy code
python fraud_detection.py
Results
The model is trained using logistic regression and evaluated on both training and testing data.

Accuracy on Training Data: [Your Training Accuracy]
Accuracy on Test Data: [Your Test Accuracy]
Conclusion
The logistic regression model provides a basic but effective approach to detecting fraudulent transactions. Further improvements can be made by exploring more advanced machine learning techniques and better handling the class imbalance in the dataset.