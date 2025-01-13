# Import necessary libraries
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Data Preprocessing

# Handle missing values before dropping columns
df['age'] = df['age'].fillna(df['age'].mean())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop columns that won't be used
df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'embarked'], inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['sex', 'pclass', 'alone'], drop_first=True)

# Define feature variables (X) and target variable (y)
X = df.drop(columns=['survived'])
y = df['survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Calculate Customer Lifetime Value (LTV)
# For the Titanic dataset, we'll use fare, number of siblings/spouses aboard (sibsp), and passenger age as proxies for average purchase value, purchase frequency, and customer lifespan
df['average_purchase_value'] = df['fare']
df['purchase_frequency'] = df['sibsp'] + 1  # Adding 1 to include the passenger's own ticket
df['customer_lifespan'] = df['age']  # Using age as a proxy for customer lifespan

# Calculate LTV
df['ltv'] = df['average_purchase_value'] * df['purchase_frequency'] * df['customer_lifespan']

# Identify high-value customers at risk of churning (not surviving in the case of Titanic)
high_value_customers = df[df['ltv'] > df['ltv'].quantile(0.75)]
at_risk_customers = high_value_customers[high_value_customers['survived'] == 0]

# Display high-value customers at risk of churning
print("\nHigh-Value Customers at Risk of Churning:")
print(at_risk_customers[['ltv', 'survived', 'average_purchase_value', 'purchase_frequency', 'customer_lifespan']])
