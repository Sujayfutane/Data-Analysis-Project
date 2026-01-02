# ----------------------------------
# STEP 1: Import required libraries
# ----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# STEP 2: Load the dataset
# ----------------------------------
df = pd.read_csv("TB_Burden_Country.csv")

print("Dataset Loaded Successfully")
print("-" * 50)

# ----------------------------------
# STEP 3: Initial Data Exploration
# ----------------------------------
print("Shape of dataset:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())

print("\nDataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

# ----------------------------------
# STEP 4: Summary Statistics
# ----------------------------------
print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))

# ----------------------------------
# STEP 5: Data Cleaning
# ----------------------------------

# 5.1 Standardize column names
df.columns = (
    df.columns
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

print("\nStandardized Column Names:")
print(df.columns.tolist())

# 5.2 Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isna().sum())

# 5.3 Fill missing numerical values with median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 5.4 Remove duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isna().sum())

# ----------------------------------
# STEP 6: Exploratory Data Analysis (EDA)
# ----------------------------------

# 6.1 Top 10 countries by TB incidence
print("\nTop 10 Countries by Estimated TB Incidence:")
print(
    df.sort_values(
        by="estimated_incidence_all_forms",
        ascending=False
    ).head(10)
)

# 6.2 Distribution of TB incidence
plt.figure()
df["estimated_incidence_all_forms"].hist(bins=30)
plt.xlabel("Estimated TB Incidence")
plt.ylabel("Frequency")
plt.title("Distribution of Estimated TB Incidence")
plt.show()

# 6.3 TB incidence vs TB mortality
plt.figure()
plt.scatter(
    df["estimated_incidence_all_forms"],
    df["estimated_mortality_tb"]
)
plt.xlabel("Estimated TB Incidence")
plt.ylabel("Estimated TB Mortality")
plt.title("TB Incidence vs Mortality")
plt.show()

# 6.4 Correlation heatmap
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 6.5 Region-wise analysis (if column exists)
if "who_region" in df.columns:
    print("\nAverage TB Incidence by WHO Region:")
    print(
        df.groupby("who_region")["estimated_incidence_all_forms"].mean()
    )

# ----------------------------------
# STEP 7: Save Cleaned Dataset
# ----------------------------------
df.to_csv("TB_Burden_Country_Cleaned.csv", index=False)
print("\nCleaned dataset saved as 'TB_Burden_Country_Cleaned.csv'")
