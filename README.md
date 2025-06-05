# stock-analytics-

🧠 Project Objective
Develop a low-risk, quarterly-updated stock selection strategy using machine learning, targeting US and German IT & Healthcare stocks, based on momentum + stability signals.

Model will label each stock as Buy (1) or Do Not Buy (0) each quarter.

# data_extraction Notebook:
🔹 Input
tickers_list: List of stock tickers (e.g., ["AAPL", "SAP.DE", ...])
start_date, end_date: Time range

🔹 Output
Pandas DataFrame (flat) with:

One row per stock per quarter
Engineered features
Target column: target (binary: 1 = Buy, 0 = Not Buy)

Some implemented features:
Data granularity	All features aligned to quarterly timestamps
Target variable	Forward 1-quarter return vs median return for classification
Leakage prevention	Lagged fundamentals only; no forward-looking data used for features
Missing data	Drop rows with >25% missing, then median-impute
Stability handling	All steps wrapped in try/except blocks per ticker
Sector + country tags	Included for filtering / strategy design
Logging / progress	Print messages per step/ticker for visibility