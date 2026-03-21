import json
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath('.'))

from src.agents.meta_manager import settle_bets

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

previous_bets = {}
previous_prices = {}

records = []

for date in dates:
    dt = pd.to_datetime(date)
    current_prices = prices_history.get(date, {})
    day_decision = decisions_history.get(date, {})
    
    # Snapshot before settlement
    caps_before = {a: agent_capital[a]['external_capital'] + agent_capital[a]['internal_capital'] for a in agents}
    
    if previous_bets and previous_prices:
        agent_capital = settle_bets(
            agent_capital,
            previous_bets,
            current_prices,
            previous_prices,
            transfer_rate=1.0,
            enable_smoothing=True
        )
        
    caps_after = {a: agent_capital[a]['external_capital'] + agent_capital[a]['internal_capital'] for a in agents}
    
    # Calculate daily changes
    changes = {a: caps_after[a] - caps_before[a] for a in agents}
    
    # Extract ROI from history if available
    rois = {a: agent_capital[a]['roi_history'][-1] if len(agent_capital[a]['roi_history']) > 0 else 0.0 for a in agents}
    
    if dt.year == 2023 and dt.month in [1, 2, 3, 4, 5]:
        records.append({
            'date': date,
            'val_loss': changes['valuation'],
            'sen_gain': changes['sentiment'],
            'val_roi': rois['valuation'],
            'sen_roi': rois['sentiment'],
            'fun_roi': rois['fundamental'],
            'avg_roi': sum(rois.values()) / 4.0 if sum(rois.values()) != 0 else 0.0
        })
        
    previous_prices.clear()
    for t in current_prices:
        previous_prices[t] = current_prices[t]
        
    for agent in agents:
        if agent in day_decision:
            previous_bets[agent] = day_decision[agent]

df = pd.DataFrame(records)
if not df.empty:
    top_val_losses = df.sort_values(by='val_loss').head(10)
    print("--- Top 10 days of Valuation Capital Loss in Early 2023 ---")
    print(top_val_losses.to_string(index=False))

print("\n--- Summary of All-Time Performance & Alpha ---")
all_dates_records = []
agent_capital = {a: {"agent_name": a, "allocated_capital": 50000.0, "external_capital": 50000.0, "internal_capital": 50000.0, "roi_history": []} for a in agents}
previous_bets = {}
previous_prices = {}

for date in dates:
    current_prices = prices_history.get(date, {})
    day_decision = decisions_history.get(date, {})
    
    if previous_bets and previous_prices:
        agent_capital = settle_bets(agent_capital, previous_bets, current_prices, previous_prices, transfer_rate=1.0, enable_smoothing=True)
        rois = {a: agent_capital[a]['roi_history'][-1] if len(agent_capital[a]['roi_history']) > 0 else 0.0 for a in agents}
        avg_roi = sum(rois.values()) / 4.0 if sum(rois.values()) != 0 else 0.0
        alphas = {a: rois[a] - avg_roi for a in agents}
        all_dates_records.append({'date': date, **{f"{a}_alpha": alphas[a] for a in agents}, **{f"{a}_roi": rois[a] for a in agents}})
        
    previous_prices.clear()
    for t in current_prices: previous_prices[t] = current_prices[t]
    for agent in agents:
        if agent in day_decision: previous_bets[agent] = day_decision[agent]

df_all = pd.DataFrame(all_dates_records)
for a in agents:
    print(f"{a.capitalize()}:")
    print(f"  Avg Daily ROI: {df_all[f'{a}_roi'].mean()*100:.4f}%")
    print(f"  Avg Daily Alpha: {df_all[f'{a}_alpha'].mean()*100:.4f}%")
    print(f"  Alpha Volatility (Std Dev): {df_all[f'{a}_alpha'].std()*100:.4f}%")
    print(f"  Positive Alpha Days Rate: {(df_all[f'{a}_alpha'] > 0).mean()*100:.2f}%") 
