# Multi-Agent System (MAS) Hedge Fund

An advanced AI-driven hedge fund architecture that leverages a **Multi-Agent System** to perform autonomous market analysis, portfolio construction, and trading. The system mimics a real-world institutional fund structure with specialized agents (Fundamental, Technical, Risk, Sentiment, Valuation) working in concert.

## 🏗 System Architecture

The architecture is divided into three main stages: **Selection (The Funnel)**, **Deliberation (The Brain)**, and **Execution (The Action)**.

---

### 1. Data Selection Layer (The "Funnel")

*Solves the "Compute Efficiency Paradox" by pre-filtering the S&P 500 down to 3-5 high-potential candidates using market physics.*

#### **Layer 1: Market Regime & Topology (The "30+30 Rule")**
*   **Regime Detection**: Uses **Minimum Spanning Tree (MST)** & **Normalized Tree Length (NTL)** to classify the market state (Contraction/Expansion).
*   **The "30+30" Candidate Pool**:
    *   **Group A (Topology)**: Selects the top 15 **Hubs** (High Degree Centrality, "Too Big to Fail") and top 15 **Leaves** (Low Degree Centrality, likely idiosyncratic movers) from the MST.
    *   **Group B (Anomalies)**: Uses **Isolation Forest** to identify the top 30 stocks with "structural breaks" (unusual price/volume behavior).
    *   **Result**: A combined universe of ~60 candidates passed to Layer 2.

#### **Layer 2: Dual-Track Scoring & Panic Detection**
*   **Clustering**: Groups filtered candidates into 5 statistically distinct clusters to ensure diversity.
*   **Panic Score**: Identifies Crash Risks using Volume + Price Physics.
    $$PanicScore = Volratio \times |Ret| \times 100$$
    *(Triggered only if $Ret < 0$)*
*   **Dual-Track Scoring**:
    Each candidate is evaluated on two tracks simultaneously:
    1.  **Long Track** (Trend Following):
        $$Score_{Long} = 0.6 \cdot Momentum + 0.4 \cdot (1 - AnomalyScore)$$
    2.  **Short Track** (Risk/Crash):
        $$Score_{Short} = 0.3 \cdot |Momentum| + 0.3 \cdot AnomalyScore + 0.2 \cdot Centrality + 0.2 \cdot PanicScore$$
*   **Final Decision**:
    *   The system compares the best *Long* vs. *Short* candidate in each cluster.
    *   **Short Condition**: If $Score_{Short} > Score_{Long} \times 1.1$, the system routes a **SHORT** task.
    *   Else, it routes a **LONG** task.

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