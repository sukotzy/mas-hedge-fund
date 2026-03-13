# dev_scripts/

开发调试脚本集合。这里存放项目开发过程中用于验证数据、测试功能、调试逻辑的一次性脚本。

**不是正式测试** — 正式 pytest 测试在 `tests/` 目录。

## 运行方式

**所有脚本必须从项目根目录运行**，否则相对路径（如 `data/raw/...`）会找不到文件：

```bash
# 正确 ✅
python dev_scripts/data/inspect_parquet.py

# 错误 ❌
cd dev_scripts/data && python inspect_parquet.py
```

## 目录结构

| 目录 | 内容 |
|---|---|
| `data/` | 检查数据文件、验证下载、探索 WRDS 数据 |
| `selection/` | 验证选股层（Layer 1 / Layer 2）输出 |
| `agents/` | 调试 allocator、对比 agent 行为、测试 settlement 逻辑 |
| `backtest/generate/` | 生成 LLM 实验数据（作为 `run_all_backtests.py` 的输入） |
| `backtest/visualize/` | 回测结果可视化 |
| `backtest/features/` | 手动计算并更新金融 features |
| `system/` | 检查 WRDS 连接、API 访问、环境配置 |
