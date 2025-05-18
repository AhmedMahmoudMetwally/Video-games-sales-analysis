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


#---------------Clean and Encoding--------------------------
# Barplot for Platform sales BEFORE cleaning
print("Number of rows before removing outliers:", len(df))
plt.figure(figsize=(12, 5))
platform_sales_before = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=platform_sales_before.values, y=platform_sales_before.index, palette='Blues_d')
plt.title('Total Global Sales by Platform (Before Cleaning)')
plt.xlabel('Total Global Sales (in millions)')
plt.ylabel('Platform')
plt.show()
# Drop rows with missing values in important columns
df = df.dropna(subset=['Year', 'Publisher', 'Genre', 'Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

# Convert 'Year' column to integer if not already
if df['Year'].dtype != 'int64':
    df['Year'] = df['Year'].astype(int)

# Remove outlier years (keep only reasonable years)
df = df[(df['Year'] >= 1980) & (df['Year'] <= 2025)]

# One-Hot Encode categorical columns
categorical_features = ['Platform', 'Genre', 'Publisher']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)


# Barplot for Platform sales AFTER cleaning
print("Number of rows after cleaning (for barplot):", len(df))
plt.figure(figsize=(12, 5))
platform_sales_after = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=platform_sales_after.values, y=platform_sales_after.index, palette='Greens_d')
plt.title('Total Global Sales by Platform (After Cleaning)')
plt.xlabel('Total Global Sales (in millions)')
plt.ylabel('Platform')
plt.show()
# Save the cleaned data to an Excel file
#df_encoded.to_excel('Vgsales-encoded.xlsx', index=False)
#print("Encoded data has been saved to 'Vgsales-encoded.xlsx'")
print(df_encoded.head())

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Sales Regions")
plt.show()

# Pairplot
print("\n=== Pairplot of Sales Data ===")
sns.pairplot(df[numeric_cols])
plt.suptitle('Pairplot of Sales Data', y=1.02)
plt.show()

# Scatter Plots
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='NA_Sales', y='Global_Sales', data=df, alpha=0.6)
plt.title('NA Sales vs Global Sales')

plt.subplot(1, 2, 2)
sns.scatterplot(x='EU_Sales', y='Global_Sales', data=df, alpha=0.6)
plt.title('EU Sales vs Global Sales')
plt.tight_layout()
plt.show()

# Analyze Relationships
print("\n=== Key Insights ===")
print("1. NA_Sales has strongest correlation with Global_Sales (0.94)")
print("2. Top platforms by sales: PS2, X360, Wii")
print("3. Action & Sports genres dominate global sales")
print("4. EU_Sales shows linear relationship with Global_Sales")
print("5. JP_Sales has weakest correlation with Global_Sales (0.62)")