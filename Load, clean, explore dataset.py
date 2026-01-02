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
print("\nColumn Names BEFORE Cleaning:\n", df.columns.tolist())

print("\nDataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

# ----------------------------------
# STEP 4: Standardize column names
# ----------------------------------
df.columns = (
    df.columns
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

print("\nColumn Names AFTER Cleaning:\n", df.columns.tolist())

# ----------------------------------
# STEP 5: Handle Missing Values & Duplicates
# ----------------------------------
print("\nMissing Values Before Cleaning:")
print(df.isna().sum())

numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isna().sum())

# ----------------------------------
# STEP 6: Automatically Detect NUMERIC Columns (FIXED)
# ----------------------------------
incidence_col = None
mortality_col = None

for col in df.columns:
    # Detect numeric incidence column
    if "incidence" in col and pd.api.types.is_numeric_dtype(df[col]):
        incidence_col = col

    # Detect numeric mortality column (exclude text/source columns)
    if "mort" in col and pd.api.types.is_numeric_dtype(df[col]):
        mortality_col = col

print("\nDetected Incidence Column:", incidence_col)
print("Detected Mortality Column:", mortality_col)

# ----------------------------------
# STEP 6A: Ensure Numeric Safety (IMPORTANT)
# ----------------------------------
if incidence_col is not None:
    df[incidence_col] = pd.to_numeric(df[incidence_col], errors="coerce")

if mortality_col is not None:
    df[mortality_col] = pd.to_numeric(df[mortality_col], errors="coerce")

# Remove rows with invalid numeric values
df = df.dropna(subset=[incidence_col, mortality_col])

# ==================================================
# STEP 7A: METRICS CALCULATION (CORE ANALYSIS)
# ==================================================
# This section calculates:
# - Mean, Median, Std Deviation
# - Skewness
# - Correlation
# - Derived Mortality Rate
# ==================================================

print("\n--- CALCULATED METRICS ---")

# Overall numeric summary
print("\nDescriptive Statistics:")
print(df.describe())

# Incidence metrics
if incidence_col is not None:
    print("\nTB Incidence Metrics:")
    print("Mean:", df[incidence_col].mean())
    print("Median:", df[incidence_col].median())
    print("Standard Deviation:", df[incidence_col].std())
    print("Skewness:", df[incidence_col].skew())

# Mortality metrics & relationships
if incidence_col is not None and mortality_col is not None:
    print("\nTB Mortality Metrics:")
    print("Mean:", df[mortality_col].mean())
    print("Median:", df[mortality_col].median())
    print("Standard Deviation:", df[mortality_col].std())

    # Correlation metric
    correlation = df[incidence_col].corr(df[mortality_col])
    print("\nCorrelation (Incidence vs Mortality):", correlation)

    # Derived metric: Mortality Rate
    df["mortality_rate"] = df[mortality_col] / df[incidence_col]

    print("\nTop 10 Countries by Mortality Rate:")
    print(df.sort_values(by="mortality_rate", ascending=False).head(10))

# ----------------------------------
# STEP 7B: Exploratory Data Analysis
# ----------------------------------

# Top 10 high-burden countries
if incidence_col is not None:
    print("\nTop 10 Countries by TB Incidence:")
    print(df.sort_values(by=incidence_col, ascending=False).head(10))

# Incidence distribution
if incidence_col is not None:
    plt.figure()
    df[incidence_col].hist(bins=30)
    plt.xlabel("TB Incidence")
    plt.ylabel("Frequency")
    plt.title("Distribution of TB Incidence")
    plt.show()

# Incidence vs Mortality scatter
if incidence_col is not None and mortality_col is not None:
    plt.figure()
    plt.scatter(df[incidence_col], df[mortality_col])
    plt.xlabel("TB Incidence")
    plt.ylabel("TB Mortality")
    plt.title("TB Incidence vs Mortality")
    plt.show()

# Correlation heatmap (visual metric)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------
# STEP 8: Region-wise Analysis
# ----------------------------------
if "who_region" in df.columns and incidence_col is not None:
    print("\nAverage TB Incidence by WHO Region:")
    print(df.groupby("who_region")[incidence_col].mean())

# ----------------------------------
# STEP 9: Save Cleaned Dataset
# ----------------------------------
df.to_csv("TB_Burden_Country_Cleaned.csv", index=False)
print("\nCleaned dataset saved successfully")
