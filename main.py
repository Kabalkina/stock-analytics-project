import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import json

from scripts.data import DataRepo
from scripts.model import Model
from scripts.simulation import StrategySimulation
from config_loader import START_DATE, END_DATE, TICKERS_LIST


# -------------------------------------------------------------------------
# Control flags
# -------------------------------------------------------------------------
LOAD_DATA = False        # If False → read quarterly_data.csv from /Data
TRAIN_MODEL = True      # If False → read rf_model_pred.csv from /Data
RUN_SIMULATION = True    # If False → read simulation.csv from /Data


# -------------------------------------------------------------------------
# Streamlit Setup
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Quarterly Stock Strategy Dashboard",
    page_icon="💼",
    layout="wide"
)


# -------------------------------------------------------------------------
# Cached Functions
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    if LOAD_DATA:
        st.info("🔄 Fetching fresh data from Yahoo Finance & FRED...")
        data = DataRepo()
        data.build_quarterly_dataset(TICKERS_LIST, START_DATE, END_DATE)
        return data.quarterly_data
    else:
        st.info("📂 Loading existing quarterly_data.csv from Data/")
        df = pd.read_csv("Data/quarterly_data.csv")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df


@st.cache_resource(show_spinner=True)
def train_model(data):
    if TRAIN_MODEL:
        st.info("🤖 Training Random Forest model...")
        model = Model(data)
        model.define_categories()
        model.get_dummies()
        model.handle_nans()
        model.split_data()
        model.train_model()
        model.make_predictions()
        model.save_data_for_simulation()
        return model
    else:
        st.info("📂 Loading existing rf_model_pred.csv from Data/")
        df = pd.read_csv("Data/rf_model_pred.csv")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # dummy wrapper so dashboard doesn’t break
        dummy = type("DummyModel", (), {})()
        dummy.df = df
        dummy.CLASSIFICATION_REPORT = {"note": "Skipped training – loaded from CSV"}
        dummy.CONFUSION_MATRIX = [[0, 0], [0, 0]]
        return dummy


@st.cache_resource(show_spinner=True)
def run_simulation(model_df):
    if RUN_SIMULATION:
        st.info("💼 Running trading simulation...")
        sim = StrategySimulation()
        history, trades, metrics = sim.backtest_fixed_capital(model_df)
        sim.save_dataset()
        return history, trades, metrics
    else:
        st.info("📂 Loading existing simulation.csv from Data/")
        history = pd.read_csv("Data/portfolio_history.csv.csv")
        trades = pd.read_csv("Data/trades.csv")
        metrics = pd.read_csv("Data/metrics.csv").to_dict(orient='records')[0]

        history['date'] = pd.to_datetime(history['date'], errors='coerce')
        trades['date'] = pd.to_datetime(trades['date'], errors='coerce')
        return history, trades, metrics

# Get last update date from metadata.json if it exists
metadata_path = os.path.join("data", "metadata.json")
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    last_update = metadata.get("last_update", "Unknown")
else:
    last_update = "Not available"
# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
#st.markdown("<div class='main-title'>📊 **Quarterly Stock Market Strategy**</div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <h1 style="text-align: center; font-size: 40px; font-weight: bold;">
        📊 Quarterly Stock Market Strategy
    </h1>
    <p style="text-align: center; font-size: 16px; color: gray;">
        Last updated: {last_update}
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    Welcome to the **Quarterly Stock Strategy Dashboard**.  
    This tool is designed for **risk-averse, long-term investors** who want a simple, data-driven approach:  
    - **Quarterly data aggregation** instead of daily noise  
    - **Random Forest model** to generate buy signals  
    - **Backtesting engine** to simulate results  

    ---
    """
)

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📈 Stock Data", "🤖 Random Forest Model", "💼 Trading Simulation"]
)

# -------------------------------------------------------------------------
# TAB 1 – STOCK DATA
# -------------------------------------------------------------------------
with tab1:
    st.markdown("<div class='sub-title'>📈 Quarterly Stock Data</div>", unsafe_allow_html=True)

    data = load_data()
    st.success(f"Loaded {len(data)} rows of {data['ticker'].nunique()} stocks")

    st.dataframe(data.head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        ticker_choice = st.selectbox("Select Stock to visualize:", sorted(data["companyName"].unique()))
        df_ticker = data[data["companyName"] == ticker_choice]

        fig, ax = plt.subplots()
        sns.lineplot(data=df_ticker, x="date", y="close", ax=ax, marker="o")
        ax.set_title(f"{ticker_choice} – Quarterly Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    with col2:
        top_momentum = data.sort_values('momentum_3q', ascending=False).drop_duplicates('ticker').head(10)
        
        fig, ax = plt.subplots()
        sns.barplot(data=top_momentum, x='momentum_3q', y='companyName', hue='sector', ax=ax)
        ax.set_title('Top 10 Stocks by 3Q Momentum')
        ax.set_xlabel('3Q Momentum')
        ax.set_ylabel('companyName')
        st.pyplot(fig)

# -------------------------------------------------------------------------
# TAB 2 – RANDOM FOREST MODEL
# -------------------------------------------------------------------------
with tab2:
    st.markdown("<div class='sub-title'>🤖 Random Forest Model Results</div>", unsafe_allow_html=True)

    model = train_model(data)
    st.success("Random Forest model trained successfully")

    col1, col2 = st.columns(2)

    # Classification Report
    with col1:
        st.subheader("📊 Classification Report")
        report = model.CLASSIFICATION_REPORT

        # Convert dict → DataFrame
        report_df = pd.DataFrame(report).T
        report_df = report_df.round(3)  # cleaner numbers

        # Highlight max values in each column (precision, recall, f1-score)
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]

        styled_report = report_df.style.apply(highlight_max, subset=['precision', 'recall', 'f1-score'])
        st.dataframe(styled_report, use_container_width=True)

    # Confusion Matrix
    with col2:
        st.subheader("🔎 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(model.CONFUSION_MATRIX, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # with col3:       
    #     st.subheader("Feature Importances: top 10")
    #     fig, ax = plt.subplots()
    #     model.FEATURE_IMPORTANCES.plot(kind='bar', ax=ax)
    #     ax.set_ylabel('Importance')

    # Current quarter buy signals
    st.subheader("📌 Recommended Stocks for Current Quarter")

    today = pd.Timestamp.today()
    current_year, current_quarter = today.year, (today.month - 1) // 3 + 1
    buy_list = model.df[
        (model.df["year"] == current_year) &
        (model.df["quarter"] == current_quarter) &
        (model.df["pred_rf"] == 1)
    ][["ticker", 'companyName', "close", "sector"]].drop_duplicates()

    if buy_list.empty:
        st.warning("No buy signals generated for this quarter.")
    else:
        st.dataframe(buy_list, use_container_width=True)
        
# -------------------------------------------------------------------------
# TAB 3 – TRADING SIMULATION
# -------------------------------------------------------------------------
with tab3:
    st.markdown("<div class='sub-title'>💼 Trading Simulation</div>", unsafe_allow_html=True)

    history, trades, metrics = run_simulation(model.df)

    # Metrics in cards
    st.subheader("📌 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Value", f"${metrics['final_value']:,.2f}")
    col2.metric("CAGR", f"{metrics['cagr']:.2f}%")
    col3.metric("Number of Trades", metrics["num_trades"])

    col1, col2 = st.columns(2)
    with col1:
        # Equity curve
        st.subheader("📈 Portfolio Value Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=history, x="date", y="portfolio_value", ax=ax, label="Portfolio Value")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Backtest Equity Curve")
        st.pyplot(fig)
    
    with col2:
        # Trades
        st.subheader("📒 Trade Log")
        st.dataframe(trades, use_container_width=True)