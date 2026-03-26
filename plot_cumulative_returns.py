import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
data_dir = Path("data/ablation_loss_aversion")
out_path = Path("plots/cumulative_returns_comparison.png")
out_path.parent.mkdir(exist_ok=True)

all_data = []

# Parse JSONL files
for file in data_dir.glob("*.jsonl"):
    name = file.stem
    
    # Extricate metadata from filename
    if "deepseek" in name:
        model = "DeepSeek"
    elif "qwen" in name:
        model = "Qwen"
    else:
        continue
        
    if "2020_crash" in name:
        period = "2020 Crash"
    elif "2022_jan" in name:
        period = "2022 Jan"
    elif "2023_svb" in name:
        period = "2023 SVB"
    else:
        continue
        
    if "with_hint" in name:
        hint = "With Hint"
    elif "no_hint" in name:
        hint = "No Hint"
        
    if "wealth" in name:
        wealth = "Wealth"
    elif "standard" in name:
        wealth = "Standard"
        
    config = f"{hint} & {wealth}"
    
    with open(file, "r", encoding="utf-8") as f:
         for line in f:
             try:
                 data = json.loads(line)
             except json.JSONDecodeError:
                 continue
             
             date = data.get("date")
             if not date: 
                 # Try finding date in day_data if nested
                 continue
                 
             val = data.get("portfolio_value")
             if val is None: 
                 continue
             
             all_data.append({
                 "Model": model,
                 "Period": period,
                 "Config": config,
                 "Date": pd.to_datetime(date),
                 "Portfolio Value": val
             })

df = pd.DataFrame(all_data)

if df.empty:
    print("❌ No data found! Check JSONL structure.")
    exit(1)

# Normalize portfolio value to cumulative returns relative to start
def normalize(group):
    group = group.sort_values("Date")
    start_val = group["Portfolio Value"].iloc[0]
    group["Cumulative Return (%)"] = (group["Portfolio Value"] / start_val - 1) * 100
    return group

df = df.groupby(["Model", "Period", "Config"], group_keys=False).apply(normalize)

# Generate Plot
sns.set_theme(style="ticks", context="paper", font_scale=1.5)
fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=False, sharey=True)

models = ["DeepSeek", "Qwen"]
periods = ["2020 Crash", "2022 Jan", "2023 SVB"]

# Globally recognized Okabe-Ito Colorblind-Safe Academic Palette
colors = {
    "No Hint & Standard": "#999999",   # Neutral Grey (Baseline)
    "No Hint & Wealth": "#56b4e9",     # Light Blue
    "With Hint & Standard": "#0072b2", # Deep Blue
    "With Hint & Wealth": "#d55e00"    # Vermillion Red (Target case)
}

line_styles = {
    "No Hint & Standard": ":",
    "No Hint & Wealth": "--",
    "With Hint & Standard": "-.",
    "With Hint & Wealth": "-"
}

for i, model in enumerate(models):
    for j, period in enumerate(periods):
        ax = axes[i, j]
        subset = df[(df["Period"] == period) & (df["Model"] == model)]
        
        if subset.empty:
            ax.set_visible(False)
            continue
            
        sns.lineplot(
            data=subset,
            x="Date",
            y="Cumulative Return (%)",
            hue="Config",
            palette=colors,
            ax=ax,
            linewidth=2.5
        )
        
        # Add horizontal line at 0%
        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
        
        # Set titles
        ax.set_title(f"{model} - {period}", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("Cum. Return (%)", fontweight="bold", fontsize=13)
        else:
            ax.set_ylabel("")
        
        # Format dates (Super compact horizontal strings to save vertical space)
        import matplotlib.dates as mdates
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) # E.g., 'Jan 15'
        ax.tick_params(axis='x', rotation=0, labelsize=11, pad=3)
        ax.tick_params(axis='y', labelsize=11, pad=3)
        
        # Manage legend (remove from individual subplots)
        if ax.legend_:
            ax.legend_.remove()
            
sns.despine(fig=fig, offset=5, trim=False)

# Single comprehensive legend at the bottom perfectly tucked in
handles, labels = axes[0, 0].get_legend_handles_labels()
if not handles:
    # Fallback if first axis doesn't have it
    handles, labels = ax.get_legend_handles_labels()
    
# Flatten the entire figure vertically using rigid coordinates 
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.17))
plt.suptitle("AI Asset Allocation Trace: Cumulative Returns over Stress Periods", fontsize=18, fontweight="bold", y=0.98)
plt.subplots_adjust(bottom=0.22, top=0.85, hspace=0.4, wspace=0.08)

plt.savefig(out_path, dpi=300) # Removed bbox_inches='tight' to respect manual compression limits
print(f"✅ Successfully saved {out_path}")
