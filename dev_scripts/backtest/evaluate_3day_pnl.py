import json
import os
import pandas as pd
from src.data.loader import LocalDataLoader
import traceback

print('Loading price data via LocalDataLoader...')
loader = LocalDataLoader()

def get_return(ticker, start_date, next_date):
    if ticker == 'CASH':
        return (0.05 / 252) # Static 5% annual RF for daily test
    try:
        prices = loader.get_prices(ticker, start_date, next_date)
        if len(prices) < 2:
            print(f"[{ticker}] NOT ENOUGH PRICES between {start_date} and {next_date}. len={len(prices)}")
            return 0.0
            
        # Ensure we are comparing the right dates
        p1 = prices[0].close
        p2 = prices[-1].close
        
        print(f"[{ticker}] {start_date}({p1}) -> {next_date}({p2})")
        if p1 == 0: return 0.0
        return (p2 - p1) / p1
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return 0.0

def eval_experiment(exp_name):
    print('\n' + '='*50)
    print(f'Experiment: {exp_name}')
    print('='*50)
    file_path = f'data/training_output_qwen_2020h1/{exp_name}/2020_01.jsonl'
    if not os.path.exists(file_path):
        print(f'{file_path} Not found')
        return
        
    dates_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            # Overwrite duplicates so we only keep the latest run's output for that date
            dates_dict[data['date']] = data
            
    # Sort dates chronologically
    dates = [dates_dict[d] for d in sorted(dates_dict.keys())]
            
    if len(dates) < 2:
        print(f'Needs >= 2 days to compute returns. Found {len(dates)}')
        return
        
    allocs = ['valuation', 'fundamental', 'technical', 'sentiment']
    caps = {k: 100.0 for k in allocs}
    
    for i in range(len(dates)-1):
        d_curr = dates[i]
        date_str = d_curr['date']
        d_next_str = dates[i+1]['date']
        
        for alloc in allocs:
            dec = d_curr.get(alloc, {})
            positions = dec.get('allocations', [])
            
            pnl = 0.0
            for pos in positions:
                tic = pos['ticker']
                amt = pos['amount'] if pos['direction'] == 'long' else -pos['amount']
                ret = get_return(tic, date_str, d_next_str)
                pnl += amt * ret
                
            caps[alloc] += pnl
            
    for k, v in caps.items():
        print(f'[{k.upper()}] Final Cap: ${v:.4f} (Ret: {((v-100.0)/100.0)*100:.4f}%)')

for e in ['no_hint_standard', 'no_hint_wealth', 'with_hint_standard', 'with_hint_wealth']:
    eval_experiment(e)
