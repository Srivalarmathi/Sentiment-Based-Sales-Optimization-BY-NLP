import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\Womens Clothing E-Commerce Reviews.csv"
df = pd.read_csv(file_path)

# Check for missing values
print("Missing values before handling:\n", df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
# or using median
# df['Age'].fillna(df['Age'].median(), inplace=True)

df['Division Name'].fillna(df['Division Name'].mode()[0], inplace=True)
df['Department Name'].fillna(df['Department Name'].mode()[0], inplace=True)
df['Class Name'].fillna(df['Class Name'].mode()[0], inplace=True)

df['Title'].fillna(df['Title'].mode()[0], inplace=True)
df['Review Text'].fillna(df['Review Text'].mode()[0], inplace=True)

# Verify the changes
print("Missing values after handling:\n", df.isnull().sum())

# Save the cleaned dataset (optional)
file_pathop = r"C:\Users\valarsri\Downloads\Womens_Clothing_Statistics_Updataed.csv"
df.to_csv(file_pathop, index=False)
