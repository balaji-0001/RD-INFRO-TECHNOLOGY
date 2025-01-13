# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Box Plots

# Box plot of fare by class
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='fare', data=df, palette='Set3')
plt.title('Fare by Class')
plt.show()

# Box plot of age by survival status
plt.figure(figsize=(10, 6))
sns.boxplot(x='survived', y='age', data=df, palette='Set2')
plt.title('Age by Survival Status')
plt.show()

# Violin Plots

# Violin plot of fare by class
plt.figure(figsize=(10, 6))
sns.violinplot(x='class', y='fare', data=df, palette='Set1')
plt.title('Fare by Class')
plt.show()

# Violin plot of age by survival status
plt.figure(figsize=(10, 6))
sns.violinplot(x='survived', y='age', data=df, palette='Set2')
plt.title('Age by Survival Status')
plt.show()

# Pair Plots

# Select relevant columns for pair plot
pair_plot_data = df[['age', 'fare', 'survived', 'pclass']].dropna()

# Pair plot
sns.pairplot(pair_plot_data, hue='survived', palette='coolwarm')
plt.show()
