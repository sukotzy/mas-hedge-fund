import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

def plot_backtest_results(input_file: str, output_image: str):
    dates = []
    portfolio_values = []

    # 1. 解析 JSONL 结果文件
    print(f"Reading data from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # 兼容不同的 key（有时叫 date, 有时叫 Date）
                date_str = data.get("date", data.get("Date"))
                value = data.get("portfolio_value", data.get("total_value"))
                
                if date_str and value is not None:
                    dates.append(date_str)
                    portfolio_values.append(float(value))
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return

    if not dates:
        print("No valid data found in the file.")
        return

    # 2. 转换为 Pandas DataFrame
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Portfolio Value': portfolio_values
    })
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # 3. 计算收益率和回撤 (和 metrics.py 逻辑一致)
    df['Cumulative Return'] = df['Portfolio Value'] / df['Portfolio Value'].iloc[0] - 1
    rolling_max = df['Portfolio Value'].cummax()
    df['Drawdown'] = (df['Portfolio Value'] - rolling_max) / rolling_max

    # 计算一些显示在图上的核心指标
    total_return = df['Cumulative Return'].iloc[-1]
    max_dd = df['Drawdown'].min()

    # 4. 开始画图 (上下双图布局)
    plt.style.use('bmh') # 使用干净的金融风格样式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'AI Hedge Fund Backtest Performance', fontsize=18, fontweight='bold')

    # --- 上方图：总净值曲线 ---
    ax1.plot(df.index, df['Portfolio Value'], color='#1f77b4', linewidth=2, label='Portfolio Value')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # 在上方图表添加文字指标框
    textstr = '\n'.join((
        f"Initial Capital: ${df['Portfolio Value'].iloc[0]:,.2f}",
        f"Final Value: ${df['Portfolio Value'].iloc[-1]:,.2f}",
        f"Total Return: {total_return:.2%}",
        f"Max Drawdown: {max_dd:.2%}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.02, 0.05, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='bottom', bbox=props)

    # --- 下方图：回撤百分比图 (Drawdown) ---
    ax2.fill_between(df.index, df['Drawdown'], 0, color='#d62728', alpha=0.3)
    ax2.plot(df.index, df['Drawdown'], color='#d62728', linewidth=1)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 优化 X 轴日期显示
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # 给主标题留点空间
    
    # 5. 保存并展示
    plt.savefig(output_image, dpi=300)
    print(f"✅ Chart saved successfully to {output_image}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot backtest results from JSONL")
    parser.add_argument("--input", type=str, required=True, help="Path to the backtest result .jsonl file")
    parser.add_argument("--output", type=str, default="backtest_chart.png", help="Path to save the generated image")
    args = parser.parse_args()
    
    plot_backtest_results(args.input, args.output)