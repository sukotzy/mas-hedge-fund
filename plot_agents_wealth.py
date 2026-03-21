import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sys
import os

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

def get_mas_wealth():
    agents = ['fundamental', 'technical', 'valuation', 'sentiment']
    smooth_file = 'data/backtests_with_risk_manager/rate005_soft_decay_segcap20_smooth.jsonl'
    
    prices_history = {}
    dates = []
    with open(smooth_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            dates.append(data['date'])
            prices_history[data['date']] = data['prices']
            
    decisions_history = {}
    decisions_dir = 'data/deepseek_standard_hint_9yr_zh/with_hint_standard'
    for file in sorted(os.listdir(decisions_dir)):
        if file.endswith('.jsonl'):
            with open(os.path.join(decisions_dir, file), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        decisions_history[data['date']] = data
                    except:
                        pass
                
    agent_capital = {
        agent: {
            "agent_name": agent,
            "allocated_capital": 50000.0,
            "external_capital": 50000.0,
            "internal_capital": 50000.0,
            "roi_history": []
        } for agent in agents
    }
    
    wealth_over_time = {agent: [] for agent in agents}
    plot_dates = []
    previous_bets = {}
    previous_prices = {}
    
    for date in dates:
        plot_dates.append(pd.to_datetime(date))
        current_prices = prices_history.get(date, {})
        day_decision = decisions_history.get(date, {})
        
        if previous_bets and previous_prices:
            agent_capital = settle_bets(
                agent_capital,
                previous_bets,
                current_prices,
                previous_prices,
                transfer_rate=1.0,
                enable_smoothing=True
            )
            
        for agent in agents:
            total_cap = agent_capital[agent]['external_capital'] + agent_capital[agent]['internal_capital']
            wealth_over_time[agent].append(total_cap)
            
        previous_prices.clear()
        for t in current_prices:
            previous_prices[t] = current_prices[t]
            
        for agent in agents:
            if agent in day_decision:
                previous_bets[agent] = day_decision[agent]
                
    return plot_dates, wealth_over_time

def main():
    print("Loading data...")
    Returns = get_individual_returns()
    mas_dates, mas_wealths = get_mas_wealth()
    
    print("Plotting figures...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    agents = ['fundamental', 'technical', 'valuation', 'sentiment']
    colors = ['blue', 'orange', 'green', 'red']
    
    # Plot 1: Individual Agents Returns
    for agent, color in zip(agents, colors):
        axs[0].plot(Returns[agent]['dates'], Returns[agent]['cum_ret'], label=agent.capitalize(), color=color)
    axs[0].set_title('Cumulative Returns of Individual Agents (segcap20 constraint)')
    axs[0].set_ylabel('Cumulative Return (Base=1.0)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: MAS Wealth Changes
    for agent, color in zip(agents, colors):
        axs[1].plot(mas_dates, mas_wealths[agent], label=f'{agent.capitalize()}', color=color)
    axs[1].set_title('Allocator Capital Adjustments inside Multi-Agent System (rate0.05_smooth)')
    axs[1].set_ylabel('Total Capital (Internal + External)')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/agents_wealth_performance.png', dpi=300)
    print("Saved to plots/agents_wealth_performance.png")

if __name__ == '__main__':
    # Ensure plots folder exists
    os.makedirs('plots', exist_ok=True)
    main()
