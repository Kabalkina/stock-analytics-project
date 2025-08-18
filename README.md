# Stock Analytics Project

Despite a lot of free available information many people have low knowledge what to do with their money despice keeping in their bank account with low interest rates. Also many are afraid to lose their money and theyfore better don't touch the money at all. My project is adressed to this group of people to see if investing not very often (quartelly) into the low or medium risk stocks could bring any value. It should be relatively safe and not that overwhelming as no need to check stock market charts every day or week and worry about making new tradying decisions. In other words, it is designed for lazy (in the good way) and risk reversive people like myself :)

🧠 Project Objective
Develop a low-risk, quarterly-updated stock selection strategy using machine learning, targeting diverse stocks around the globe (America, Europe and Asia-Pasific regions), based on momentum + stability signals.

Model will label each stock as buy signal (1 for "yes" and 0 for "no") each quarter.

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