# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Calculate and Visualize Overall Survival Rate
survival_rate = df['survived'].value_counts(normalize=True)

plt.figure(figsize=(6, 6))
plt.pie(survival_rate, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Overall Survival Rate')
plt.show()

# Explore Passenger Distribution by Various Demographics

# Visualize gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', data=df, palette='Set2')
plt.title('Passenger Distribution by Gender')
plt.show()

# Visualize age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'].dropna(), bins=30, kde=True, color='purple')
plt.title('Passenger Age Distribution')
plt.show()

# Analyze Fare Distribution

plt.figure(figsize=(10, 6))
sns.histplot(df['fare'], bins=30, kde=True, color='green')
plt.title('Passenger Fare Distribution')
plt.show()

# Investigate Relationships Between Survival and Different Classes/Embarkation Points

# Survival by class
plt.figure(figsize=(10, 6))
sns.countplot(x='class', hue='survived', data=df, palette='Set1')
plt.title('Survival by Class')
plt.show()

# Survival by embarkation point
plt.figure(figsize=(10, 6))
sns.countplot(x='embarked', hue='survived', data=df, palette='Set3')
plt.title('Survival by Embarkation Point')
plt.show()
