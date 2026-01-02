# ==================================================
# TB BURDEN DATA ANALYSIS - INTERACTIVE DASHBOARD
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

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

# Prepare top 15 for readable barplots
top_incidence = df.sort_values(by=incidence_col, ascending=False).head(15)
top_mortality = df.sort_values(by="mortality_rate_pct", ascending=False).head(15)

# -------------------------
# Create 2x3 Interactive Dashboard
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(26, 15))
plt.suptitle("🌟 TB Burden Interactive Dashboard", fontsize=28, fontweight='bold', color="#2F4F4F")

# ---- 1️⃣ Top 15 TB Incidence ----
bars1 = axes[0, 0].barh(top_incidence[country_col][::-1], top_incidence[incidence_col][::-1], color="#1f77b4", alpha=0.8)
axes[0, 0].set_title("Top 15 Countries by TB Incidence", fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel("Incidence")
axes[0, 0].set_ylabel("Country")
# Interactive hover
cursor1 = mplcursors.cursor(bars1)
cursor1.connect("add", lambda sel: sel.annotation.set_text(f"{top_incidence[incidence_col][::-1].iloc[sel.index]:,.0f}"))

# ---- 2️⃣ Top 15 Mortality Rate (%) ----
bars2 = axes[0, 1].barh(top_mortality[country_col][::-1], top_mortality["mortality_rate_pct"][::-1], color="#d62728", alpha=0.8)
axes[0, 1].set_title("Top 15 Countries by Mortality Rate (%)", fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel("Mortality Rate (%)")
axes[0, 1].set_ylabel("Country")
cursor2 = mplcursors.cursor(bars2)
cursor2.connect("add", lambda sel: sel.annotation.set_text(f"{top_mortality['mortality_rate_pct'][::-1].iloc[sel.index]:.2f}%"))

# ---- 3️⃣ Histogram of TB Incidence ----
n, bins, patches = axes[0, 2].hist(df[incidence_col], bins=30, color="#1E90FF", alpha=0.6, edgecolor='black')
axes[0, 2].set_title("Distribution of TB Incidence", fontsize=16, fontweight='bold')
axes[0, 2].set_xlabel("TB Incidence")
axes[0, 2].set_ylabel("Frequency")
# Hover over histogram bins
cursor3 = mplcursors.cursor(patches)
cursor3.connect("add", lambda sel: sel.annotation.set_text(f"Count: {int(sel.artist.get_height())}"))

# ---- 4️⃣ Scatter: Incidence vs Mortality ----
scatter_colors = []
if "who_region" in df.columns:
    regions = df["who_region"].unique()
    cmap = plt.cm.tab10
    for i, region in enumerate(regions):
        subset = df[df["who_region"] == region]
        sc = axes[1, 0].scatter(subset[incidence_col], subset[mortality_col], s=100, alpha=0.7, edgecolor="black", color=cmap(i % 10), label=region)
        scatter_colors.append(sc)
    axes[1, 0].legend(title="WHO Region", bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    sc = axes[1, 0].scatter(df[incidence_col], df[mortality_col], s=100, alpha=0.7, edgecolor="black", color="#FF8C00")
    scatter_colors.append(sc)

axes[1, 0].set_title("TB Incidence vs Mortality", fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel("Incidence")
axes[1, 0].set_ylabel("Mortality")

# Hover over scatter points
for sc in scatter_colors:
    cursor_sc = mplcursors.cursor(sc)
    cursor_sc.connect("add", lambda sel: sel.annotation.set_text(
        f"{country_col}: {df.iloc[sel.index][country_col]}\nIncidence: {df.iloc[sel.index][incidence_col]:,.0f}\nMortality: {df.iloc[sel.index][mortality_col]:,.0f}"
    ))

# ---- 5️⃣ Correlation Heatmap ----
corr = df[[incidence_col, mortality_col, "mortality_rate_pct"]].corr()
im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_xticks(np.arange(len(corr.columns)))
axes[1, 1].set_yticks(np.arange(len(corr.index)))
axes[1, 1].set_xticklabels(corr.columns)
axes[1, 1].set_yticklabels(corr.index)
axes[1, 1].set_title("Correlation Heatmap (Key Metrics)", fontsize=16, fontweight='bold')
# Annotate correlation values
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

# ---- 6️⃣ Average TB Incidence by WHO Region ----
if "who_region" in df.columns:
    region_stats = df.groupby("who_region")[incidence_col].mean().sort_values(ascending=False)
    bars6 = axes[1, 2].barh(region_stats.index[::-1], region_stats.values[::-1],
                            color=plt.cm.viridis(np.linspace(0,1,len(region_stats))), alpha=0.8)
    axes[1, 2].set_title("Average TB Incidence by WHO Region", fontsize=16, fontweight='bold')
    axes[1, 2].set_xlabel("Average Incidence")
    axes[1, 2].set_ylabel("WHO Region")
    # Hover over region bars
    cursor6 = mplcursors.cursor(bars6)
    cursor6.connect("add", lambda sel: sel.annotation.set_text(f"{region_stats.values[::-1][sel.index]:,.0f}"))
else:
    axes[1, 2].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -------------------------
# Save cleaned dataset
# -------------------------
df.to_csv("TB_Burden_Country_Cleaned.csv", index=False)
print("✅ Cleaned dataset saved successfully")
