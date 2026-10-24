# House Price Predictor (Beginner ML Project)

A small Streamlit web app using Python, pandas, and scikit-learn. It trains a Random Forest regression model on the California Housing dataset and predicts a district-level median house value.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the dataset, so an internet connection is needed.

## What this project demonstrates
- Loading a real dataset
- Train/test split
- Regression with `RandomForestRegressor`
- Model evaluation with MAE and R²
- Interactive predictions through Streamlit

## Important limitation
This dataset predicts a district's median house value in California, in units of $100,000. It does **not** estimate the market price of a specific house or property in India. Do not present the demo as a real-estate valuation system.
