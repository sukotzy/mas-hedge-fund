# Multi-Agent System (MAS) Hedge Fund

An advanced AI-driven hedge fund architecture that leverages a **Multi-Agent System** to perform autonomous market analysis, portfolio construction, and trading. The system mimics a real-world institutional fund structure with specialized agents (Fundamental, Technical, Risk, Sentiment, Valuation) working in concert.

## 🏗 System Architecture

The architecture is divided into three main stages: **Selection (The Funnel)**, **Deliberation (The Brain)**, and **Execution (The Action)**.

---

### 1. Data Selection Layer (The "Funnel")

*Solves the "Compute Efficiency Paradox" by pre-filtering the S&P 500 down to 3-5 high-potential candidates using market physics.*

#### **Layer 1: Market Regime & Macro Pre-Screening**
*   **Market Physics**: Uses **Minimum Spanning Tree (MST)** & **Normalized Tree Length (NTL)**.
    *   **Contraction (Crisis)**: Detected when the MST shrinks significantly (Z-Score < -1.5 vs rolling history), indicating high system-wide correlation and panic.
    *   **Expansion (Normal)**: Detected when MST expands, indicating a healthy market driven by idiosyncratic factors.
*   **Short Selling Engine**: Uses **Isolation Forest** (Unsupervised Anomaly Detection).
    *   Identifies stocks with "structural breaks" (e.g., price/volume divergence, abnormal volatility) as potential candidates for short selling (Forensic/Valuation targets).

#### **Layer 2: Diversity & Candidate Selection**
*   **Clustering**: Groups stocks into statistically distinct clusters (via Hierarchical Risk Parity / Ward's Linkage) to ensure portfolio orthogonality (diversity).
*   **Selection Logic**:
    *   **Long Scenario**: Selects the highest momentum stock in the cluster.
    *   **Short Scenario**: Selects the stock with the highest Anomaly Score if it exceeds the threshold (0.0).

#### **🧪 A/B Testing: The "Hint" Toggle**
To benchmark the reasoning capabilities of the AI agents, the selection layer supports two modes:
1.  **With Hint (Default)** (`include_hint=True`): The system explicitly passes the recommended direction (`Long` or `Short`) and the specific reason (e.g., "High Anomaly Score") to the agents.
2.  **Blind Mode** (`include_hint=False`): The system provides only a neutral `Analyze` instruction. This forces the agents to derive the direction purely from their own analysis, allowing you to test if they can independently identify the opportunity/risk.

---

### 2. Multi-Agent Deliberation (The "Brain")

*Selected assets are analyzed by a committee of specialized "Asset Allocators" (LLM Agents).*

#### **Diverse Analyst Personas**
Each agent specializes in a distinct source of alpha:
*   **Fundamental Analyst**: Deep dives into financial statements, moats, and competitive advantage (Buffett-style).
*   **Technical Analyst**: Analyzes price action, trends, and support/resistance levels.
*   **Valuation Analyst**: Focuses on DCF models and intrinsic value gaps.
*   **Sentiment Analyst**: Gauges market psychology from news and social signals.

#### **🧪 A/B Testing: Prompt Strategies**
The agents can be initialized with different cognitive prompts to test behavioral economics theories:
1.  **Standard Prompt**: The agent simply analyzes the asset to maximize accuracy.
2.  **Wealth Impact Prompt** ("Skin in the Game"): The agent is explicitly told that **their own internal capital**—and thus their future influence in the fund—depends on the outcome of this specific bet. This tests if "financial survival instinct" improves decision quality.

#### **Prediction Market Mechanism**
Instead of a simple "Vote", agents participate in an internal **Betting Market**.
*   **Belief as Capital**: Agents invest their allocated capital into outcomes: `Up`, `Neutral`, or `Down`.
*   **Conviction Sizing**: High conviction = Larger bet.
*   **Fair Price Discovery**: The aggregation of all agent bets forms a "Market Implied Fair Price" for the asset, synthesizing all diverse views into a single signal.

#### **Dual-Tranche Capital System**
Agents are incentivized through two capital pools:
1.  **Internal Capital (Meritocratic)**: Agents accumulate capital by making winning bets. Successful agents gain more influence (voting power) over time.
2.  **External Capital (Top-Down)**: Allocated by the **Meta Manager**.
    *   **Meta Manager**: A supervisor agent that dynamically shifts capital based on the Market Regime (e.g., funding the *Short Seller* and *Risk Manager* during a Crisis, or the *Technical Analyst* during a Bull Run). *(Note: Currently implemented rule-based, training in progress)*.

---

### 3. Execution & Risk (The "Action")

*   **Portfolio Construction**: The final "Fair Price" from the betting market is fed into an optimizer.
*   **Risk Constraints**: The optimizer strictly adheres to volatility limits, leverage caps, and sector exposure limits before generating final trade orders.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- WRDS Account (for institutional quality data)
- OpenAI API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/mas-hedge-fund.git

# Install dependencies
pip install -r requirements.txt
```

### Running the Selection Pipeline

```bash
# Run verification script with a specific date
python -m src.testing_scripts.test_selection_layer --date 2023-01-04
```