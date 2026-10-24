import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Predictor")
st.write("A beginner-friendly ML demo that predicts a California district's median house value.")
st.caption("Educational demo only. It is not suitable for estimating real property prices.")

@st.cache_resource
def train_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    target = "MedHouseVal"
    X = df.drop(columns=[target])
    y = df[target]  # target is measured in units of $100,000

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "feature_names": list(X.columns),
        "data": data,
    }
    return model, metrics

with st.spinner("Loading the dataset and training the model (first run may take a moment)..."):
    try:
        model, metrics = train_model()
    except Exception as exc:
        st.error("Could not load the California Housing dataset. Check your internet connection and try again.")
        st.exception(exc)
        st.stop()

st.subheader("Enter house-area characteristics")
data = metrics["data"]
feature_descriptions = {
    "MedInc": "Median income in the district (tens of thousands of USD)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "District population",
    "AveOccup": "Average people per household",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

defaults = {
    "MedInc": 5.0, "HouseAge": 25.0, "AveRooms": 5.0, "AveBedrms": 1.0,
    "Population": 1000.0, "AveOccup": 3.0, "Latitude": 34.05, "Longitude": -118.25
}
limits = {
    "MedInc": (0.5, 15.0), "HouseAge": (1.0, 52.0), "AveRooms": (1.0, 20.0),
    "AveBedrms": (0.2, 5.0), "Population": (1.0, 40000.0), "AveOccup": (1.0, 20.0),
    "Latitude": (32.5, 42.0), "Longitude": (-124.5, -114.0)
}
values = {}
for feature in metrics["feature_names"]:
    low, high = limits[feature]
    values[feature] = st.number_input(
        feature_descriptions[feature],
        min_value=float(low),
        max_value=float(high),
        value=float(defaults[feature]),
        step=0.1 if feature not in ("Population",) else 100.0,
        help=f"Dataset feature: {feature}"
    )

if st.button("Predict house value", type="primary", use_container_width=True):
    row = pd.DataFrame([values], columns=metrics["feature_names"])
    prediction = float(model.predict(row)[0])
    estimated_usd = prediction * 100_000
    st.success(f"Estimated median value: ${estimated_usd:,.0f}")
    st.write("This is the dataset's estimated **median house value for a district**, not an individual property's market price.")

st.divider()
st.subheader("Model evaluation")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (test set)", f"${metrics['mae'] * 100_000:,.0f}")
col2.metric("R² score (test set)", f"{metrics['r2']:.3f}")
st.write(
    "**How to interpret:** MAE is the average absolute prediction error on the held-out test set. "
    "R² describes how much variation the model explains; 1.0 is a perfect fit. "
    "These metrics are from a simple Random Forest model and may vary by setup."
)
st.caption("Dataset: scikit-learn California Housing dataset. Features describe districts, not individual homes.")
