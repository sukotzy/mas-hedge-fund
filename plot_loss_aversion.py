import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_jsonl(filepath):
    dates = []
    gross_exposures = []
    portfolio_values = []
    net_exposures = []

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                
                ptf_val = data.get("portfolio_value", 0.0)
                prices = data.get("prices", {})
                holdings = data.get("updated_holdings", {})
                
                gross_exp = 0.0
                net_exp = 0.0
                
                for ticker, pos in holdings.items():
                    price = prices.get(ticker, 0.0)
                    if price > 0:
                        gross_exp += (pos["long"] + pos["short"]) * price
                        net_exp += (pos["long"] - pos["short"]) * price
                
                dates.append(data["date"])
                gross_exposures.append(gross_exp)
                net_exposures.append(net_exp)
                portfolio_values.append(ptf_val)
                
            except Exception as e:
                print(f"Error parsing line in {filepath}: {e}")

    if not dates:
        return None
        
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Gross_Exposure': gross_exposures,
        'Net_Exposure': net_exposures,
        'Portfolio_Value': portfolio_values
    })
    
    # Calculate Ratios
    # Cash = Portfolio Value - Net Exposure (since margin = 0.0)
    df['Cash'] = df['Portfolio_Value'] - df['Net_Exposure']
    
    df['Gross_Exposure_Ratio'] = df['Gross_Exposure'] / df['Portfolio_Value']
    df['Cash_Ratio'] = df['Cash'] / df['Portfolio_Value']
    
    return {
        'avg_gross_exposure_ratio': df['Gross_Exposure_Ratio'].mean(),
        'avg_cash_ratio': df['Cash_Ratio'].mean(),
    }

def main():
    data_dir = "data/ablation_loss_aversion"
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    if not files:
        print(f"No jsonl files found in {data_dir}. Are they still running?")
        return
        
    records = []
    for f in files:
        filename = os.path.basename(f)
        # Expected e.g.: data_allocator_reuslts_test_results_deepseek_2020_crash_with_hint_standard.jsonl
        # Let's extract metadata based on string splits
        name_no_ext = filename.replace(".jsonl", "")
        parts = name_no_ext.split("_")
        
        # We need to find the model, period, hint, and allocator
        # Typical pattern: ..._results_deepseek_2020_crash_with_hint_standard
        # Or: ..._results_qwen_2023_svb_no_hint_wealth
        try:
            res_idx = parts.index("results")
            model = parts[res_idx + 1] # deepseek or qwen
            
            period_str = parts[res_idx + 2] + "_" + parts[res_idx + 3] # e.g. 2020_crash, 2022_jan, 2023_svb
            
            hint_str = "with_hint" if "with_hint" in name_no_ext else "no_hint"
            allocator_str = "wealth" if "wealth" in name_no_ext else "standard"
            
            metrics = analyze_jsonl(f)
            if metrics:
                records.append({
                    "Model": model.capitalize(),
                    "Period": period_str.replace("_", " ").title(),
                    "Setting": f"{hint_str.replace('_', ' ')} & {allocator_str}".title(),
                    "Gross Exposure (%)": metrics["avg_gross_exposure_ratio"] * 100,
                    "Cash Allocation (%)": metrics["avg_cash_ratio"] * 100
                })
        except Exception as e:
            print(f"Error extracting metadata from {filename}: {e}")
            
    if not records:
        print("No valid records extracted.")
        return
        
    df_results = pd.DataFrame(records)
    print("\n--- Aggregated Loss Aversion Data ---")
    print(df_results.to_string())
    
    # Save the dataframe
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(os.path.join(output_dir, "loss_aversion_metrics.csv"), index=False)
    
    # Plotting!
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    # Globally recognized Okabe-Ito Colorblind-Safe Academic Palette
    colors = {
        "No Hint & Standard": "#999999",   # Neutral Grey (Baseline)
        "No Hint & Wealth": "#56b4e9",     # Light Blue
        "With Hint & Standard": "#0072b2", # Deep Blue
        "With Hint & Wealth": "#d55e00"    # Vermillion Red (Target case)
    }
    
    # Combined Horizontal Layout (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.2), sharey=False)
    
    plot_configs = [
        ("Deepseek", "Gross Exposure (%)"),
        ("Deepseek", "Cash Allocation (%)"),
        ("Qwen", "Gross Exposure (%)"),
        ("Qwen", "Cash Allocation (%)")
    ]
    
    for idx, (model, metric) in enumerate(plot_configs):
        ax = axes[idx]
        subset = df_results[df_results["Model"] == model]
        
        if subset.empty:
            continue
            
        sns.barplot(
            data=subset, x="Period", y=metric, hue="Setting",
            palette=colors, ax=ax, alpha=0.9, err_kws={'linewidth': 1.5}
        )
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontweight="bold")
        
        # simplified title to save vertical space
        ax.set_title(f"{model} - {metric.split(' ')[0]}", fontweight="bold", fontsize=15)
            
        if ax.legend_:
            ax.legend_.remove()

    # Move legend outside tight layout
    sns.despine(trim=True)
    
    # Removed suptitle for publication figure array formatting
    
    # Absolute strict mathematical positioning mapping for 1x4 horizontal strips
    plt.subplots_adjust(bottom=0.20, top=0.92, left=0.05, right=0.98, wspace=0.15)
    
    handles, labels = axes[0].get_legend_handles_labels()
    # Place legend exactly 5% below the bottom axes margin
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.15), fontsize=15)
    
    combined_path = os.path.join(output_dir, "loss_aversion_combined.png")
    # Remove bbox_inches='tight' so it respects the rigid subplots_adjust dimensions layout
    plt.savefig(combined_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Successfully generated Loss Aversion combined 1x4 visual chart: {combined_path}")

if __name__ == "__main__":
    main()
