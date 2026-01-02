# ==================================================
# TB BURDEN DATA ANALYSIS - PROFESSIONAL DASHBOARD
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted")

# -------------------------
# Load and clean dataset
# -------------------------
df = pd.read_csv("TB_Burden_Country.csv")

# Clean column names
df.columns = df.columns.str.lower().str.replace(r"[^\w]", "_", regex=True)

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with "Unknown"
categorical_cols = df.select_dtypes(exclude=np.number).columns
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# Drop duplicates
df = df.drop_duplicates()

# -------------------------
# Detect key columns
# -------------------------
country_col = next((c for c in df.columns if "country" in c), None)
incidence_col = next((c for c in df.columns if "incidence" in c and pd.api.types.is_numeric_dtype(df[c])), None)
mortality_col = next((c for c in df.columns if "mort" in c and pd.api.types.is_numeric_dtype(df[c])), None)

df[incidence_col] = pd.to_numeric(df[incidence_col], errors="coerce")
df[mortality_col] = pd.to_numeric(df[mortality_col], errors="coerce")
df = df.dropna(subset=[incidence_col, mortality_col])

# Derived metrics
df["mortality_rate"] = df[mortality_col] / df[incidence_col]
df["mortality_rate_pct"] = df["mortality_rate"] * 100

# -------------------------
# Prepare Top 15 for Barplots (Readable)
# -------------------------
top_incidence = df.sort_values(by=incidence_col, ascending=False).head(15)
top_mortality = df.sort_values(by="mortality_rate_pct", ascending=False).head(15)

# -------------------------
# Create 2x3 Dashboard
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(26, 15))
plt.suptitle("🌟 TB Burden Analysis Dashboard", fontsize=28, fontweight='bold', color="#2F4F4F")

# ---- 1️⃣ Top 15 TB Incidence ----
sns.barplot(
    data=top_incidence,
    y=country_col,
    x=incidence_col,
    palette="Blues_r",
    alpha=0.9,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Top 15 Countries by TB Incidence", fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel("Incidence")
axes[0, 0].set_ylabel("Country")

# ---- 2️⃣ Top 15 Mortality Rate (%) ----
sns.barplot(
    data=top_mortality,
    y=country_col,
    x="mortality_rate_pct",
    palette="Reds_r",
    alpha=0.9,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Top 15 Countries by Mortality Rate (%)", fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel("Mortality Rate (%)")
axes[0, 1].set_ylabel("Country")

# ---- 3️⃣ Histogram of TB Incidence ----
sns.histplot(
    df[incidence_col],
    bins=30,
    kde=True,
    color="#1E90FF",
    alpha=0.6,
    ax=axes[0, 2]
)
axes[0, 2].set_title("Distribution of TB Incidence", fontsize=16, fontweight='bold')
axes[0, 2].set_xlabel("TB Incidence")
axes[0, 2].set_ylabel("Frequency")

# ---- 4️⃣ Scatter: Incidence vs Mortality ----
if "who_region" in df.columns:
    sns.scatterplot(
        data=df,
        x=incidence_col,
        y=mortality_col,
        hue="who_region",
        palette="Set2",
        s=120,
        alpha=0.8,
        edgecolor="black",
        ax=axes[1, 0]
    )
    axes[1, 0].legend(title="WHO Region", bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    sns.scatterplot(
        data=df,
        x=incidence_col,
        y=mortality_col,
        color="#FF8C00",
        s=120,
        alpha=0.8,
        ax=axes[1, 0]
    )
axes[1, 0].set_title("TB Incidence vs Mortality", fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel("Incidence")
axes[1, 0].set_ylabel("Mortality")

# ---- 5️⃣ Correlation Heatmap ----
key_cols = [incidence_col, mortality_col, "mortality_rate_pct"]
sns.heatmap(
    df[key_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=1,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Correlation Heatmap (Key Metrics)", fontsize=16, fontweight='bold')

# ---- 6️⃣ Average TB Incidence by WHO Region ----
if "who_region" in df.columns:
    region_stats = df.groupby("who_region")[incidence_col].mean().sort_values(ascending=False).reset_index()
    sns.barplot(
        data=region_stats,
        x=incidence_col,
        y="who_region",
        palette="viridis",
        alpha=0.9,
        ax=axes[1, 2]
    )
    axes[1, 2].set_title("Average TB Incidence by WHO Region", fontsize=16, fontweight='bold')
    axes[1, 2].set_xlabel("Average Incidence")
    axes[1, 2].set_ylabel("WHO Region")
else:
    axes[1, 2].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -------------------------
# Save cleaned dataset
# -------------------------
df.to_csv("TB_Burden_Country_Cleaned.csv", index=False)
print("✅ Cleaned dataset saved successfully")
