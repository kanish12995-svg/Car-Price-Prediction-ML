
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

st.title("🚘 Car Price Prediction Result")
st.write("Your Machine Learning Prediction Report")

st.divider()

# Check input
if "car_details" not in st.session_state:
    st.warning("Please enter car details first.")

    if st.button("Back to Home"):
        st.switch_page("home.py")

    st.stop()

# Load trained model
model_path = BASE_DIR / "car_model.pkl"

if not model_path.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

saved = joblib.load(model_path)

model = saved["model"]
r2 = saved["r2"]
mae = saved["mae"]
df = saved["data"]
features = saved["features"]

# Get input
details = st.session_state["car_details"]

input_data = pd.DataFrame(
    [details],
    columns=features
)

# Predict price
predicted_price = float(
    model.predict(input_data)[0]
)

# Individual tree predictions
preprocessor = model.named_steps["preprocessor"]
forest = model.named_steps["regressor"]

transformed = preprocessor.transform(input_data)

tree_predictions = np.array([
    tree.predict(transformed)[0]
    for tree in forest.estimators_
])

minimum_price = float(
    np.percentile(tree_predictions, 10)
)

maximum_price = float(
    np.percentile(tree_predictions, 90)
)

# Car details
st.subheader("🚗 Selected Car Details")

c1, c2 = st.columns(2)

c1.write("**Company:** " + details["Company_Name"])
c1.write("**Brand:** " + details["Brand"])

c2.write("**Model Year:** " + str(details["Model"]))
c2.write("**Condition:** " + details["Condition"])

st.divider()

# Predicted price
st.subheader("💰 Estimated Selling Price")

st.metric(
    "Predicted Car Price",
    f"₹ {predicted_price:,.0f}"
)

st.divider()

# Model performance
st.subheader("📊 Model Performance")

c1, c2 = st.columns(2)

c1.metric("R² Score", f"{r2 * 100:.2f}%")
c2.metric("Mean Absolute Error", f"₹ {mae:,.0f}")

st.caption(
    "R² measures variation explained on test data. "
    "MAE measures average absolute prediction error."
)

st.divider()

# Price range
st.subheader("📈 Estimated Price Range")

c1, c2 = st.columns(2)

c1.metric("Minimum Estimate", f"₹ {minimum_price:,.0f}")
c2.metric("Maximum Estimate", f"₹ {maximum_price:,.0f}")

st.caption(
    "These estimates use the 10th and 90th percentiles "
    "of individual tree predictions. They are not guaranteed "
    "market-price confidence intervals."
)

st.divider()

# Price comparison graph
st.subheader("📊 Price Comparison Graph")

company_data = df[
    df["Company_Name"] == details["Company_Name"]
]

average_price = company_data["Car_Price"].mean()

fig, ax = plt.subplots(figsize=(9, 4))

labels = ["Company Average", "Predicted Price"]
values = [average_price, predicted_price]

bars = ax.bar(labels, values)

ax.set_title("Company Average vs Predicted Price")
ax.set_ylabel("Price (₹)")

ax.bar_label(
    bars,
    labels=[f"₹ {v:,.0f}" for v in values],
    padding=3
)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.divider()

# Prediction details
st.subheader("🔍 Prediction Details")

st.write("**Algorithm:** Random Forest Regressor")
st.write("**Number of Trees:** 200")
st.write("**Training Split:** 80%")
st.write("**Testing Split:** 20%")
st.write("**Input Features:** Company, Brand, Model, Condition")
st.write("**Target:** Car_Price")

st.divider()

# Download report
st.subheader("📥 Download Prediction Report")

report = pd.DataFrame([{
    "Company": details["Company_Name"],
    "Brand": details["Brand"],
    "Model": details["Model"],
    "Condition": details["Condition"],
    "Predicted_Price": round(predicted_price, 2),
    "Minimum_Estimate": round(minimum_price, 2),
    "Maximum_Estimate": round(maximum_price, 2),
    "R2_Score": round(r2, 4),
    "MAE": round(mae, 2),
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
}])

st.download_button(
    "⬇ Download Prediction Report",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name="Car_Prediction_Report.csv",
    mime="text/csv",
    use_container_width=True
)

st.divider()

if st.button("⬅ Back to Home", use_container_width=True):
    st.switch_page("home.py")



