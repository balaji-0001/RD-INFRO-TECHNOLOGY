# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values
df['age'].fillna(df['age'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.drop(columns=['deck'], inplace=True)
df.dropna(subset=['embark_town'], inplace=True)

# Check for missing values again to confirm they are filled
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# One-Hot Encoding
categorical_columns = ['sex', 'embarked', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Display the processed dataframe
print("\nProcessed DataFrame:")
print(df_encoded.head())

# Visualize Data with Seaborn

# Bar Plot of survival rate by class
plt.figure(figsize=(10, 6))
sns.barplot(x='class', y='survived', data=df)
plt.title('Survival Rate by Class')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
