# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Segment Passengers based on fare, age, and class
df['fare_segment'] = pd.cut(df['fare'], bins=[0, 25, 50, 75, 100, 600], labels=['0-25', '25-50', '50-75', '75-100', '100+'])
df['age_segment'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 50, 80], labels=['0-12', '12-18', '18-35', '35-50', '50+'])

# Display the segmented data
print("Segmented Data:")
print(df[['fare_segment', 'age_segment', 'class']].head())

# Analyze Survival Rates Across Segments

# Calculate survival rates for each fare segment
survival_rate_fare = df.groupby('fare_segment')['survived'].mean().reset_index()

# Calculate survival rates for each age segment
survival_rate_age = df.groupby('age_segment')['survived'].mean().reset_index()

# Calculate survival rates for each class
survival_rate_class = df.groupby('class')['survived'].mean().reset_index()

# Display the survival rates
print("Survival Rate by Fare Segment:")
print(survival_rate_fare)
print("\nSurvival Rate by Age Segment:")
print(survival_rate_age)
print("\nSurvival Rate by Class:")
print(survival_rate_class)

# Visualize Survival Rates

# Visualize survival rate by fare segment
plt.figure(figsize=(10, 6))
sns.barplot(x='fare_segment', y='survived', data=survival_rate_fare, palette='Blues')
plt.title('Survival Rate by Fare Segment')
plt.show()

# Visualize survival rate by age segment
plt.figure(figsize=(10, 6))
sns.barplot(x='age_segment', y='survived', data=survival_rate_age, palette='Greens')
plt.title('Survival Rate by Age Segment')
plt.show()

# Visualize survival rate by class
plt.figure(figsize=(10, 6))
sns.barplot(x='class', y='survived', data=survival_rate_class, palette='Reds')
plt.title('Survival Rate by Class')
plt.show()
