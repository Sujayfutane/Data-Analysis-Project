# ==================================================
# TB BURDEN ANALYSIS - INTERACTIVE DASHBOARD (PLOTLY)
# ==================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Step 1: Load dataset
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
# Step 2: Detect key columns
# -------------------------
country_col = next((c for c in df.columns if "country" in c), None)
incidence_col = next((c for c in df.columns if "incidence" in c and pd.api.types.is_numeric_dtype(df[c])), None)
mortality_col = next((c for c in df.columns if "mort" in c and pd.api.types.is_numeric_dtype(df[c])), None)

# Ensure numeric
df[incidence_col] = pd.to_numeric(df[incidence_col], errors="coerce")
df[mortality_col] = pd.to_numeric(df[mortality_col], errors="coerce")
df = df.dropna(subset=[incidence_col, mortality_col])

# Derived metrics
df["mortality_rate"] = df[mortality_col] / df[incidence_col]
df["mortality_rate_pct"] = df["mortality_rate"] * 100

# Region stats if exists
if "who_region" in df.columns:
    region_stats = df.groupby("who_region")[incidence_col].mean().sort_values(ascending=False).reset_index()

# -------------------------
# Step 3: Create interactive dashboard
# -------------------------
fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=(
        "Top 15 Countries by TB Incidence",
        "Top 15 Countries by Mortality Rate (%)",
        "Distribution of TB Incidence",
        "TB Incidence vs Mortality",
        "Correlation Heatmap (Key Metrics)",
        "Average TB Incidence by WHO Region",
        "Insights Panel",
        ""
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
           [{"type": "heatmap"}, {"type": "bar"}, {"type": "table"}, None]]
)

# ---- Top 15 TB Incidence ----
top_incidence = df.sort_values(by=incidence_col, ascending=False).head(15)
fig.add_trace(
    go.Bar(
        x=top_incidence[incidence_col][::-1],
        y=top_incidence[country_col][::-1],
        orientation='h',
        marker_color=px.colors.sequential.Blues,
        hovertemplate='%{y}: %{x:,}'
    ),
    row=1, col=1
)

# ---- Top 15 Mortality Rate (%) ----
top_mortality = df.sort_values(by="mortality_rate_pct", ascending=False).head(15)
fig.add_trace(
    go.Bar(
        x=top_mortality["mortality_rate_pct"][::-1],
        y=top_mortality[country_col][::-1],
        orientation='h',
        marker_color=px.colors.sequential.Reds,
        hovertemplate='%{y}: %{x:.2f}%'
    ),
    row=1, col=2
)

# ---- Histogram of TB Incidence ----
fig.add_trace(
    go.Histogram(
        x=df[incidence_col],
        nbinsx=30,
        marker_color="#1E90FF",
        opacity=0.7,
        hovertemplate='Incidence: %{x:,}<br>Count: %{y}'
    ),
    row=1, col=3
)

# ---- Scatter: Incidence vs Mortality ----
fig.add_trace(
    go.Scatter(
        x=df[incidence_col],
        y=df[mortality_col],
        mode='markers',
        marker=dict(size=12, color="#FF8C00", line=dict(width=1, color='black')),
        text=df[country_col],  # used by hovertemplate
        hovertemplate='%{text}<br>Incidence: %{x:,}<br>Mortality: %{y:,}'
    ),
    row=1, col=4
)

# Highlight top 5% mortality
high_mort = df[df["mortality_rate_pct"] > df["mortality_rate_pct"].quantile(0.95)]
fig.add_trace(
    go.Scatter(
        x=high_mort[incidence_col],
        y=high_mort[mortality_col],
        mode='markers',
        marker=dict(size=14, color='red', symbol='diamond'),
        name='Top 5% Mortality',
        text=high_mort[country_col],
        hovertemplate='%{text}<br>Incidence: %{x:,}<br>Mortality: %{y:,}'
    ),
    row=1, col=4
)

# ---- Correlation Heatmap ----
corr_matrix = df[[incidence_col, mortality_col, "mortality_rate_pct"]].corr()
fig.add_trace(
    go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        hovertemplate='Metric 1: %{y}<br>Metric 2: %{x}<br>Correlation: %{z:.2f}'
    ),
    row=2, col=1
)

# ---- Region-wise Average TB Incidence ----
if "who_region" in df.columns:
    fig.add_trace(
        go.Bar(
            x=region_stats[incidence_col][::-1],
            y=region_stats["who_region"][::-1],
            orientation='h',
            marker_color=px.colors.sequential.Viridis,
            hovertemplate='%{y}: %{x:,.0f}'
        ),
        row=2, col=2
    )

# ---- Insights Panel ----
top5_mortality = top_mortality.head(5)
insights_text = f"""
Key Metrics:
Mean Incidence: {df[incidence_col].mean():.0f}
Median Incidence: {df[incidence_col].median():.0f}
Mean Mortality: {df[mortality_col].mean():.0f}
Median Mortality: {df[mortality_col].median():.0f}
Incidence vs Mortality Corr: {df[incidence_col].corr(df[mortality_col]):.2f}

Countries with highest mortality rate:
{', '.join(top5_mortality[country_col].tolist())}
"""
fig.add_trace(
    go.Table(
        header=dict(values=["Insights"], fill_color="#4B0082", font=dict(color='white', size=14)),
        cells=dict(values=[[insights_text]], align="left", fill_color="#E6E6FA", font=dict(color="#2F4F4F", size=12))
    ),
    row=2, col=3
)

# ---- Layout ----
fig.update_layout(
    height=900,
    width=1800,
    title_text="🌍 TB Burden Interactive Dashboard",
    title_font_size=28,
    showlegend=False
)

fig.show()

# -------------------------
# Step 4: Save cleaned dataset
# -------------------------
df.to_csv("TB_Burden_Country_Cleaned.csv", index=False)
print("✅ Cleaned dataset saved: TB_Burden_Country_Cleaned.csv")
