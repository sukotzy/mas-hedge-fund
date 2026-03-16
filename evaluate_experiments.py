import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器环境使用 Agg
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Dict
from src.data.loader import get_local_loader

def load_backtest_data(filepath: str) -> pd.Series:
    """读取 JSONL 返回带有日期的 Series"""
    dates = []
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            date_str = data.get("date", data.get("Date"))
            value = data.get("portfolio_value", data.get("total_value"))
            if date_str and value is not None:
                dates.append(date_str)
                values.append(float(value))
    
    df = pd.DataFrame({'Value': values}, index=pd.to_datetime(dates))
    df = df.groupby(df.index).last()  # 去重
    df.sort_index(inplace=True)
    return df['Value']

def calculate_metrics(series: pd.Series, annual_rf_rate=0.0434) -> dict:
    """计算专业的量化评估指标"""
    returns = series.pct_change().dropna()
    if len(returns) < 2:
        return {}

    trading_days = 252
    
    # 累计收益率
    cum_return = (series.iloc[-1] / series.iloc[0]) - 1
    
    # 年化收益率 (CAGR)
    days_passed = (series.index[-1] - series.index[0]).days
    years_passed = days_passed / 365.25
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years_passed) - 1 if years_passed > 0 else 0

    # 年化波动率
    ann_volatility = returns.std() * np.sqrt(trading_days)
    
    # 夏普比率
    daily_rf = annual_rf_rate / trading_days
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(trading_days) if returns.std() > 0 else 0

    # 索提诺比率
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(trading_days) if downside_std > 0 else float('inf')

    # 最大回撤
    rolling_max = series.cummax()
    drawdowns = (series - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # 卡玛比率 (Calmar Ratio)
    calmar = cagr / abs(max_dd) if max_dd < 0 else float('inf')

    return {
        "Initial Value": series.iloc[0],
        "Final Value": series.iloc[-1],
        "Total Return": cum_return,
        "CAGR": cagr,
        "Ann. Volatility": ann_volatility,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar
    }

def main():
    parser = argparse.ArgumentParser(description="Professional Backtest Evaluator")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of backtest .jsonl files")
    parser.add_argument("--labels", nargs='+', required=True, help="List of names for the legends")
    parser.add_argument("--output", default="strategy_comparison.png", help="Output image file")
    parser.add_argument("--benchmark", default="SPY", help="Local ticker for benchmark (e.g., SPY)")
    parser.add_argument("--title", default="AI Hedge Fund Strategy Comparison", help="Main title of the plot")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        print("Error: Number of inputs must match number of labels.")
        return

    # 1. 加载所有策略数据并对齐
    df_dict = {}
    for filepath, label in zip(args.inputs, args.labels):
        print(f"Loading {label} from {filepath}...")
        df_dict[label] = load_backtest_data(filepath)
    
    df_all = pd.DataFrame(df_dict)
    df_all.ffill(inplace=True)  # 前向填充缺失值
    df_all.dropna(inplace=True) # 丢弃开始对不齐的部分
    
    start_date = df_all.index.min()
    end_date = df_all.index.max()

    # 2. 下载并对齐基准指数 (S&P 500)
    print(f"Loading Benchmark {args.benchmark} from local parquet from {start_date.date()} to {end_date.date()}...")
    try:
        loader = get_local_loader("data")
        bench_df = loader.get_ticker_data(args.benchmark, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if not bench_df.empty:
            bench_series = bench_df.set_index('date')['prc']
            
            # 将 Benchmark 重置为与策略相同的初始资金
            initial_capital = df_all.iloc[0, 0] 
            
            # 对齐索引（以策略的交易日为准），前向填充缺失值
            bench_aligned = bench_series.reindex(df_all.index).ffill()
            
            # 归一化并放大
            df_all['S&P 500 (Bench)'] = (bench_aligned / bench_aligned.iloc[0]) * initial_capital
        else:
            print(f"Warning: No local data found for benchmark {args.benchmark}")
    except Exception as e:
        print(f"Warning: Could not fetch local benchmark. Error: {e}")

    # 3. 计算所有指标
    print("Calculating metrics...")
    metrics_list = {}
    for col in df_all.columns:
        metrics_list[col] = calculate_metrics(df_all[col])
    
    metrics_df = pd.DataFrame(metrics_list)

    # 4. 开始画图
    plt.style.use('bmh') # 专业金融风格
    # 创建布局：上图占 5 份（净值），中图占 2 份（回撤），下图占 3 份（表格）
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 2, 3], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    fig.suptitle(args.title, fontsize=20, fontweight='bold', y=0.95)

    # 巧妙处理 X 轴，消除周末断层：使用整数索引作图
    x_indices = np.arange(len(df_all))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    if 'S&P 500 (Bench)' in df_all.columns:
        # 让基准变成虚线和灰色
        colors[len(df_all.columns)-1] = '#7f7f7f'

    # --- Panel 1: 净值曲线 ---
    for i, col in enumerate(df_all.columns):
        linestyle = '--' if 'Bench' in col else '-'
        linewidth = 2.5 if 'Bench' not in col else 1.5
        alpha = 0.7 if 'Bench' in col else 1.0
        ax1.plot(x_indices, df_all[col], label=col, color=colors[i], 
                 linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    ax1.set_title('Cumulative Portfolio Value', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 2: 最大回撤曲线 ---
    rolling_max_all = df_all.cummax()
    drawdowns_all = (df_all - rolling_max_all) / rolling_max_all
    
    for i, col in enumerate(drawdowns_all.columns):
        linestyle = '--' if 'Bench' in col else '-'
        linewidth = 1.5 if 'Bench' not in col else 1.0
        ax2.plot(x_indices, drawdowns_all[col], label=col, color=colors[i], 
                 linestyle=linestyle, linewidth=linewidth)
        if 'Bench' not in col:
            ax2.fill_between(x_indices, drawdowns_all[col], 0, color=colors[i], alpha=0.1)

    ax2.set_title('Underwater Plot (Drawdowns)', fontsize=14)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Adaptive X-Axis Ticks: Ensure consistent intervals and show all dates for short backtests

    # 统一设置 X 轴标签
    if len(df_all) <= 20:
        # 天数较少，显示每一天，确保间隔和视觉上的均匀
        tick_indices = np.arange(len(df_all))
    else:
        # 天数较多，使用均匀步长，避免日期跳跃感不一致
        num_ticks = 10
        tick_indices = np.linspace(0, len(df_all) - 1, num_ticks, dtype=int)
    
    tick_labels = [df_all.index[i].strftime('%Y-%m-%d') for i in tick_indices]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_xlim(0, len(df_all)-1)

    # --- Panel 3: 业绩评估指标表格 ---
    ax3.axis('off')
    
    # 格式化表格数据
    # Cast to object to avoid FutureWarning when setting string values in numeric columns
    formatted_df = metrics_df.copy().astype(object)
    for col in formatted_df.columns:
        formatted_df.loc['Initial Value', col] = f"${formatted_df.loc['Initial Value', col]:,.2f}"
        formatted_df.loc['Final Value', col] = f"${formatted_df.loc['Final Value', col]:,.2f}"
        formatted_df.loc['Total Return', col] = f"{formatted_df.loc['Total Return', col]:.2%}"
        formatted_df.loc['CAGR', col] = f"{formatted_df.loc['CAGR', col]:.2%}"
        formatted_df.loc['Ann. Volatility', col] = f"{formatted_df.loc['Ann. Volatility', col]:.2%}"
        formatted_df.loc['Max Drawdown', col] = f"{formatted_df.loc['Max Drawdown', col]:.2%}"
        formatted_df.loc['Sharpe Ratio', col] = f"{formatted_df.loc['Sharpe Ratio', col]:.2f}"
        formatted_df.loc['Sortino Ratio', col] = f"{formatted_df.loc['Sortino Ratio', col]:.2f}"
        formatted_df.loc['Calmar Ratio', col] = f"{formatted_df.loc['Calmar Ratio', col]:.2f}"

    # 绘制表格
    table = ax3.table(cellText=formatted_df.values,
                      rowLabels=formatted_df.index,
                      colLabels=formatted_df.columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0.05, 0.1, 0.9, 0.8])
    
    # 美化表格
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif j == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f9f9f9')

    # 保存图片
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Professional comparison chart saved to: {args.output}")

if __name__ == "__main__":
    main()