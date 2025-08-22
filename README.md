# 📈 Quarterly Stock Market Strategy


**A data-driven exploration of whether “lazy” quarterly rebalancing can beat the bank — using Random Forest models, macro indicators, and trading simulations.**  

**Technologies:** Python · Random Forest Model · Streamlit · GitHub Actions


## 🌟 Introduction

Despite the abundance of freely available financial information, many people still leave their savings idle in bank accounts earning minimal interest.
For a large group of **risk-averse** investors, the fear of losing money — combined with the complexity and stress of daily market monitoring — discourages them from investing altogether.

This project asks a simple question:
👉 *Can “lazy” investors, who only rebalance their portfolio a few times a year, still achieve meaningful returns without the pressure of daily trading?*

To test this idea, I analyzed **55 stocks across America, Europe, and Asia-Pacific** using data from Yahoo Finance covering the period **Jan 2020 – Aug 2025**. The full stock list can be found in `config.yaml`. To enrich the dataset, I also incorporated macro-financial indicators from FRED.

At the core of the project is a **Random Forest classification model** that predicts whether a stock should be bought in a given quarter (`buy_signal = 1`). The model was trained and evaluated, and then a simulation was run to test the strategy in practice.

* **Initial capital**: $7,000
* **Transaction fee**: $2
* **Trading horizon (simulated)**: ~25 years of historical data
* **Risk management**: stop-loss at -20% and take-profit at 30%
* **Results**: Final portfolio value of **~$84k**, with a CAGR of **~11%**

These results suggest that a **low-frequency, rule-based strategy** can offer an appealing balance between risk and return — especially for investors who don’t want to spend their lives glued to stock charts.

You can explore the project interactively in the **[Streamlit Dashboard](https://stockmarketproject.streamlit.app/)**, where you’ll find:

* Stock price visualizations
* Random Forest model performance
* Trading strategy simulation results


![Dashboard Preview](images/dashboard.png)


This project was developed as part of the **DataTalksClub -- Stock Market Analytics course**.

------------------------------------------------------------------------

## 🏗️ Project Structure
  
    project-root/
    │
    ├── main.py                 # Main entry point – runs the full pipeline and Streamlit dashboard app
    ├── automation.py           # Monthly automation script (GitHub Actions entrypoint)
    ├── .github/workflows/      # GitHub Actions automation (monthly updates)
    ├── config.yaml             # User configuration (dates, tickers, parameters)
    ├── config_loader.py        # Loads config.yaml for use in scripts
    ├── requirements.txt        # Python dependencies
    ├── .gitignore              # Git ignore file
    │
    ├── notebooks/              # Initial exploration & prototyping
    │   ├── data_extraction.ipynb
    │   ├── model.ipynb
    │   └── simulation.ipynb
    │
    ├── scripts/                # Core logic (productionized from notebooks)
    │   ├── data.py             # Data extraction & feature engineering
    │   ├── model.py            # Machine learning pipeline (Random Forest)
    │   └── simulation.py       # Backtesting and portfolio simulation
    │
    ├── images/                 # Dashboard screenshots
    └── data/                   # Output data & results (created at runtime)
        ├── quarterly_data.csv
        ├── rf_model_pred.csv
        ├── portfolio_history.csv
        ├── trades.csv
        └── metrics.json

------------------------------------------------------------------------

## ⚙️ Automation (GitHub Actions)

The project includes **monthly automation** to:  
- Run pipeline on the **last day of the month**  
- Refresh data (if `LOAD_DATA=True`)  
- Retrain model (if `TRAIN_MODEL=True`)  
- Run backtest & save updated results  

Workflow defined in: `.github/workflows/monthly_update.yml`  

### Example run log:
```
LOAD_DATA=False | TRAIN_MODEL=False | RUN_SIMULATION=True
Loaded quarterly_data.csv (cached)
Skipped model training (using rf_model_pred.csv)
Backtest executed → updated portfolio_history.csv, trades.csv, metrics.csv
```

------------------------------------------------------------------------

## ⚙️ Workflow

The project runs as a **three-stage pipeline**:

1.  **Data Preparation** (`scripts/data.py`)
    -   Extracts **price features** (quarterly aggregates of daily stock
        data from Yahoo Finance)\
    -   Pulls **fundamental features** (valuation ratios, debt metrics,
        ESG, earnings)\
    -   Joins with **macro indicators** (FRED series, market ETFs)\
    -   Saves a **quarterly dataset** to `Data/quarterly_data.csv`
2.  **Model Training & Prediction** (`scripts/model.py`)
    -   Defines categorical variables & target (`buy_signal`)\
    -   Encodes categorical features (tickers, sectors)\
    -   Splits data chronologically (80% train / 20% test)\
    -   Trains a **RandomForestClassifier** with **TimeSeriesSplit
        cross-validation**\
    -   Saves best model and creates predictions →
        `Data/rf_model_pred.csv`
3.  **Trading Simulation** (`scripts/simulation.py`)
    -   Runs a backtest with fixed capital allocation\
    -   Applies stop-loss and take-profit thresholds\
    -   Computes metrics: **final value, CAGR, number of trades**\
    -   Optionally tunes parameters over a grid of stop-loss /
        take-profit combinations\
    -   Saves portfolio history and trades → `Data/simulation.csv`

The **main pipeline** (`main.py`) runs all three steps automatically.

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set up environment

``` bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

### 3. Configure

Edit (if needed) **`config.yaml`** to set: - `START_DATE`, `END_DATE` (historical
data range)\
- `TICKERS_LIST` (portfolio tickers)\
- `INITIAL_CAPITAL`, `TRANSACTION_FEE`, `STOP_LOSS_PCT`,
`TAKE_PROFIT_PCT`

### 4. Run pipeline

``` bash
streamlit run main.py
```
### 5. GitHub Actions (optional)

This workflow runs automatically on schedule.  
If you'd like to run manually:

```bash
python automation.py
```


This will: - Download and process financial data\
- Train a Random Forest model\
- Run trading simulation\
- Save all results into the `Data/` folder\
- Generate a Streamlit Dashboard with visuals and KPIs
- Run an automated update

If Yahoo Finance doesn't work, the data in csv format is stored in Google drive [here](https://drive.google.com/drive/folders/1fdJtoJGsMp1IoJdFj1Ru2AscUCTpaPqh?usp=sharing)

------------------------------------------------------------------------

## 📊 Example Outputs

-   `quarterly_data.csv`: Engineered dataset (features per
    ticker/quarter)\
-   `rf_model_pred.csv`: Model predictions (buy/hold signals)\
-   `portfolio_history.csv`: Portfolio performance history
-   `trades.csv`: Trades done bae´sed on the trading strategy
-   `metrics.json`: Trading strategy performance

------------------------------------------------------------------------

## 💡 Notes

-   **Logging:** All scripts use Python's `logging` module for
    structured output.\
-   **Reproducibility:** Raw data comes from Yahoo Finance & FRED APIs
    --- results may differ if re-run at different times.\
-   **Notebooks:** The original exploratory analysis is preserved in
    `notebooks/`, while production code lives in `scripts/`.
