import os
import sys
import subprocess
import argparse
from pathlib import Path

# 确保环境变量正确
os.environ["USE_LOCAL_DATA"] = "true"
os.environ["PYTHONPATH"] = "."

def main():
    parser = argparse.ArgumentParser(description="Batch Backtesting Engine (Continuous Compounding Mode)")
    parser.add_argument("--base-dirs", nargs='+', default=["data/allocator_reuslts_test"], 
                        help="Root directories to scan for experiment results (.jsonl folders)")
    parser.add_argument("--out-dir", type=str, default="data/backtests_with_risk_manager",
                        help="Folder to save backtest results")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial portfolio cash")
    parser.add_argument("--margin-requirement", type=float, default=0.25, help="Margin requirement")
    parser.add_argument("--fast", action="store_true", help="Use pre-loaded PriceMatrix for fast O(1) lookups")
    parser.add_argument("--agent", type=str, choices=["fundamental", "technical", "valuation", "sentiment"], default=None, help="Run single agent ablation study for a specific agent")
    parser.add_argument("--disable-risk-manager", action="store_true", help="Disable Risk Manager and allow full allocations")
    parser.add_argument("--turnover-penalty", type=float, default=0.05, help="L1 penalty for turnover in QP optimizer")
    parser.add_argument("--decay-mode", type=str, choices=["none", "soft", "harsh"], default="harsh", help="Configure the kinematic decay speed for legacy positions")
    parser.add_argument("--segregate-capital", type=float, default=0.0, help="Ratio (0.0 to 1.0) of capital to allocate to fresh signals vs old decayed holdings. 0.0 disables segregation.")
    parser.add_argument("--transfer-rate", type=float, default=1.0, help="Multiplier for the zero-sum capital transfer penalty/reward.")
    parser.add_argument("--enable-smoothing", action="store_true", help="Enable EMA smoothing for alpha and capital floor protection in the betting market.")
    parser.add_argument("--enable-safety-net", action="store_true", help="Enable maximum daily loss caps and absolute bankruptcy floors in the betting market.")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.25, help="Maximum percentage of current capital an agent can lose in a single day during zero-sum settlement.")
    
    args = parser.parse_args()

    # 创建输出文件夹
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("🚀 Starting Batch Backtesting Engine (Continuous Compounding Mode) 🚀")
    if args.agent:
        print(f"🔬 RUNNING SINGLE-AGENT ABLATION STUDY: {args.agent.upper()} 🔬")
    print("=========================================================================")

    # 需要扫描的实验结果根目录
    base_dirs = args.base_dirs
    experiment_dirs = set()

    # 1. 寻找所有包含 .jsonl 文件的"实验文件夹"
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist. Skipping.")
            continue
        for root, dirs, files in os.walk(base_dir):
            # 如果当前文件夹下有 .jsonl 文件，说明这是一个实验的子文件夹
            if any(f.endswith('.jsonl') for f in files):
                experiment_dirs.add(root)

    if not experiment_dirs:
        print("⚠️ 警告: 没有在指定目录下找到包含 .jsonl 的实验结果文件夹！")
        return


    temp_file = Path("temp_combined_experiment.jsonl")

    # 2. 遍历每一个实验组合
    for exp_dir in sorted(experiment_dirs):
        print("\n=====================================================")
        print(f"📁 Processing Experiment Directory: {exp_dir}")
        
        # 将路径转换为安全的文件名
        safe_name = str(exp_dir).replace("data\\", "").replace("data/", "").replace("\\", "_").replace("/", "_")
        
        # 如果是单跑 Agent，在输出文件名加上 agent 前缀
        if args.agent:
            output_file = out_dir / f"{safe_name}_{args.agent}.jsonl"
        else:
            output_file = out_dir / f"{safe_name}.jsonl"
        
        # 获取该文件夹下所有的 .jsonl 文件，并按字母(时间)顺序排序
        jsonl_files = sorted(Path(exp_dir).glob("*.jsonl"))
        
        if not jsonl_files:
            continue

        # 3. 按月份顺序拼接所有文件，保证时间线连续
        with open(temp_file, "w", encoding="utf-8") as outfile:
            for jf in jsonl_files:
                with open(jf, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)
                    if content and not content.endswith('\n'):
                        outfile.write('\n')
                        
        print(f"✅ Combined {len(jsonl_files)} monthly files. Starting backtest...")
        
        hf_python = Path("hf/Scripts/python.exe")
        python_exe = str(hf_python.resolve()) if hf_python.exists() else sys.executable
        
        # 4. 调用回测引擎
        if args.agent:
            cmd = [
                python_exe, "run_single_agent_backtest.py",
                "--agent", args.agent,
                "--input-file", str(temp_file),
                "--output-file", str(output_file),
                "--initial-cash", str(args.initial_cash),
                "--margin-requirement", str(args.margin_requirement)
            ]
        else:
            cmd = [
                python_exe, "src/training/run_optimization.py",
                "--input-file", str(temp_file),
                "--output-file", str(output_file),
                "--initial-cash", str(args.initial_cash),
                "--margin-requirement", str(args.margin_requirement)
            ]
        
        if args.fast:
            cmd.append("--fast")
        if args.disable_risk_manager:
            cmd.append("--disable-risk-manager")
        if args.turnover_penalty != 0.05:
            cmd.extend(["--turnover-penalty", str(args.turnover_penalty)])
        if args.decay_mode != "harsh":
            cmd.extend(["--decay-mode", args.decay_mode])
        if args.segregate_capital > 0.0:
            cmd.extend(["--segregate-capital", str(args.segregate_capital)])
        if args.transfer_rate != 1.0:
            cmd.extend(["--transfer-rate", str(args.transfer_rate)])
        if args.enable_smoothing:
            cmd.append("--enable-smoothing")
        if args.enable_safety_net:
            cmd.append("--enable-safety-net")
        if args.max_daily_loss_pct != 0.25:
            cmd.extend(["--max-daily-loss-pct", str(args.max_daily_loss_pct)])
        
        try:
            subprocess.run(cmd, check=True)
            print(f"🎉 Finished -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running backtest for {exp_dir}: {e}")

    # 5. 打扫战场
    if temp_file.exists():
        temp_file.unlink()

    print("=========================================================================")
    print(f"🏆 All backtests completed! Results saved in {out_dir}/")

if __name__ == "__main__":
    main()