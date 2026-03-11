import os
import sys
import subprocess
from pathlib import Path

# 确保环境变量正确
os.environ["USE_LOCAL_DATA"] = "true"
os.environ["PYTHONPATH"] = "."

# 创建输出文件夹
out_dir = Path("data/backtests")
out_dir.mkdir(parents=True, exist_ok=True)

print("=========================================================================")
print("🚀 Starting Batch Backtesting Engine (Continuous Compounding Mode) 🚀")
print("=========================================================================")

# 需要扫描的实验结果根目录
base_dirs = ["data/allocator_reuslts_test"]
experiment_dirs = set()

# 1. 寻找所有包含 .jsonl 文件的“实验文件夹”
for base_dir in base_dirs:
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Please create it and add the deepseek and qwen result folders.")
        continue
    for root, dirs, files in os.walk(base_dir):
        # 如果当前文件夹下有 .jsonl 文件，说明这是一个实验的子文件夹 (例如 2020_crash, 2022_jan 等)
        if any(f.endswith('.jsonl') for f in files):
            experiment_dirs.add(root)

if not experiment_dirs:
    print("⚠️ 警告: 没有在 data/ 目录下找到包含 .jsonl 的实验结果文件夹！")
    print("请确认你已经运行了实验（例如 run_deepseek_experiments.bat）并成功生成了输出文件。")
    exit(0)


temp_file = Path("temp_combined_experiment.jsonl")

# 2. 遍历每一个实验组合
for exp_dir in sorted(experiment_dirs):
    print("\n=====================================================")
    print(f"📁 Processing Experiment Directory: {exp_dir}")
    
    # 将路径转换为安全的文件名 (例如: results_deepseek_2020_crash_no_hint)
    # 兼容 Windows 的反斜杠 \ 和 Mac/Linux 的正斜杠 /
    safe_name = str(exp_dir).replace("data\\", "").replace("data/", "").replace("\\", "_").replace("/", "_")
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
                # 确保拼接处有换行符，防止 JSON 解析报错
                if content and not content.endswith('\n'):
                    outfile.write('\n')
                    
    print(f"✅ Combined {len(jsonl_files)} monthly files into one timeline. Starting backtest...")
    
    # 获取当前项目下的 hf 环境的 python 路径
    hf_python = Path("hf/Scripts/python.exe")
    python_exe = str(hf_python.resolve()) if hf_python.exists() else sys.executable
    
    # 4. 调用回测引擎
    cmd = [
        python_exe, "src/training/run_optimization.py",
        "--input-file", str(temp_file),
        "--output-file", str(output_file),
        "--initial-cash", "100000.0",
        "--margin-requirement", "0.5"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"🎉 Finished -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running backtest for {exp_dir}: {e}")

# 5. 打扫战场，删除临时文件
if temp_file.exists():
    temp_file.unlink()

print("=========================================================================")
print(f"🏆 All continuous backtests completed successfully! Results saved in {out_dir}/")