# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
try:
    df = pd.read_csv('vgsales.csv')
    print("Dataset loaded successfully with shape:", df.shape)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Display basic information
print("\n=== Dataset Information ===")
print(df.info())

# Display first 5 rows
print("\n=== First 5 Rows ===")
print(df.head())

# Data Types Description
print("\n=== Data Types ===")
print(df.dtypes)

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\nNumerical Columns:", numerical_cols.tolist())
print("Categorical Columns:", categorical_cols.tolist())

# Basic Statistics
print("\n=== Basic Statistics for Numerical Columns ===")
print(df.describe())

print("\n=== Statistics for Categorical Columns ===")
print(df[categorical_cols].describe())

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Target Variable Analysis (Global_Sales)
print("\n=== Target Variable Analysis ===")
print("Global Sales Summary Statistics:")
print(df['Global_Sales'].describe())

# Distribution of Global Sales
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Global_Sales'], bins=50, kde=True)
plt.title('Distribution of Global Sales')
plt.xlabel('Global Sales (in millions)')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Global_Sales'])
plt.title('Boxplot of Global Sales')
plt.xlabel('Global Sales (in millions)')

plt.tight_layout()
plt.show()

# Log transformation to handle skewness
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(np.log1p(df['Global_Sales']), bins=50, kde=True)
plt.title('Log-Transformed Global Sales')
plt.xlabel('log(Global Sales + 1)')

plt.subplot(1, 2, 2)
sns.boxplot(x=np.log1p(df['Global_Sales']))
plt.title('Boxplot of Log-Transformed Sales')
plt.xlabel('log(Global Sales + 1)')

plt.tight_layout()
plt.show()

# Top selling games
print("\n=== Top 10 Best-Selling Games ===")
print(df[['Name', 'Platform', 'Year', 'Global_Sales']]
      .sort_values('Global_Sales', ascending=False)
      .head(10)
      .reset_index(drop=True))

# Sales by Platform
plt.figure(figsize=(12, 8))
platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=platform_sales.values, y=platform_sales.index, palette='viridis')
plt.title('Total Global Sales by Platform')
plt.xlabel('Total Global Sales (in millions)')
plt.ylabel('Platform')
plt.show()

# Sales by Genre
plt.figure(figsize=(12, 8))
genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=genre_sales.values, y=genre_sales.index, palette='magma')
plt.title('Total Global Sales by Genre')
plt.xlabel('Total Global Sales (in millions)')
plt.ylabel('Genre')
plt.show()

# Sales over time
plt.figure(figsize=(14, 6))
yearly_sales = df.groupby('Year')['Global_Sales'].sum()
sns.lineplot(x=yearly_sales.index, y=yearly_sales.values)
plt.title('Global Video Game Sales Over Time')
plt.xlabel('Year')
plt.ylabel('Total Global Sales (in millions)')
plt.xticks(rotation=45)
plt.show()

# Correlation between sales regions
print("\n=== Correlation Between Sales Regions ===")
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
print(df[sales_cols].corr())

plt.figure(figsize=(10, 8))
sns.heatmap(df[sales_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Sales Data')
plt.show()