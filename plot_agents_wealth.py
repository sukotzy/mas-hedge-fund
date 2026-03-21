import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sys
import os
import argparse

sys.path.append(os.path.abspath('.'))

from src.agents.meta_manager import settle_bets

def get_individual_returns():
    agents = ['fundamental', 'technical', 'valuation', 'sentiment']
    base_path = 'data/backtests_with_risk_manager'
    returns = {}
    
    for agent in agents:
        file = f'{base_path}/rate005_soft_decay_segcap20_{agent}.jsonl'
        dates = []
        portfolios = []
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                dates.append(pd.to_datetime(data['date']))
                portfolios.append(data['portfolio_value'])
                
        cum_ret = [p / portfolios[0] for p in portfolios]
        returns[agent] = {'dates': dates, 'cum_ret': cum_ret}
        
    return returns

def get_mas_wealth(smooth_file):
    agents = ['fundamental', 'technical', 'valuation', 'sentiment', 'virtual_cash']
    
    wealth_over_time = {agent: [] for agent in agents}
    plot_dates = []
    
    with open(smooth_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            plot_dates.append(pd.to_datetime(data['date']))
            
            # Extract directly from the generated JSONL
            agent_cap_dict = data.get('agent_capital', {})
            for agent in agents:
                if agent in agent_cap_dict:
                    total_cap = agent_cap_dict[agent].get('external_capital', 0) + agent_cap_dict[agent].get('internal_capital', 0)
                else:
                    total_cap = 100000.0  # Default initial capital (50k internal + 50k external)
                wealth_over_time[agent].append(total_cap)
                
    return plot_dates, wealth_over_time

def main():
    parser = argparse.ArgumentParser(description="Plot MAS wealth trajectories.")
    parser.add_argument("--input", "-i", type=str, default="data/backtests_with_risk_manager/rate005_soft_decay_segcap20_smooth_eta50_tau005.jsonl", help="The combined MAS JSONL output file.")
    parser.add_argument("--out", "-o", type=str, default="plots/agents_wealth_performance.png", help="Output PNG path.")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    Returns = get_individual_returns()
    mas_dates, mas_wealths = get_mas_wealth(args.input)
    
    print("Plotting figures...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    agents_all = ['fundamental', 'technical', 'valuation', 'sentiment', 'virtual_cash']
    colors_all = ['blue', 'orange', 'green', 'red', 'black']
    
    agents = ['fundamental', 'technical', 'valuation', 'sentiment']
    colors = ['blue', 'orange', 'green', 'red']
    
    # Plot 1: Individual Agents Returns (Exclude virtual_cash since it doesn't have a single run jsonl)
    for agent, color in zip(agents, colors):
        try:
            axs[0].plot(Returns[agent]['dates'], Returns[agent]['cum_ret'], label=agent.capitalize(), color=color)
        except KeyError:
            pass
    axs[0].set_title('Cumulative Returns of Individual Agents (segcap20 constraint)')
    axs[0].set_ylabel('Cumulative Return (Base=1.0)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: MAS Wealth Changes (Includes virtual_cash)
    for agent, color in zip(agents_all, colors_all):
        axs[1].plot(mas_dates, mas_wealths[agent], label=f'{agent.capitalize()}', color=color)
    axs[1].set_title('Allocator Capital Adjustments inside Multi-Agent System (rate0.05_smooth_eta50_tau005)')
    axs[1].set_ylabel('Total Capital (Internal + External)')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved to {args.out}")

if __name__ == '__main__':
    # Ensure plots folder exists
    os.makedirs('plots', exist_ok=True)
    main()
