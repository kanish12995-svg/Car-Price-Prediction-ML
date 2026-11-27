import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.title("🚗  Old Car Price Prediction System")
st.subheader("Used Car Price Prediction Using Machine Learning")

st.write(
    "Enter your car details to estimate its selling price."
)

# Load dataset
df = pd.read_csv(BASE_DIR / "car_data.csv")

st.divider()
st.header("Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    # Company Dropdown
    company = st.selectbox(
        "Select Company",
        sorted(df["Company_Name"].unique())
    )

    # Filter dataset according to selected company
    company_data = df[df["Company_Name"] == company]

    # Brand Dropdown (Only selected company's brands)
    brand = st.selectbox(
        "Select Brand",
        sorted(company_data["Brand"].unique())
    )

with col2:

    # Filter according to company and brand
    car_data = company_data[company_data["Brand"] == brand]

    # Model Year Dropdown
    model_year = st.selectbox(
        "Select Model Year",
        sorted(car_data["Model"].unique())
    )

    # Condition Dropdown
    condition = st.selectbox(
        "Select Condition",
        sorted(car_data["Condition"].unique())
    )

st.divider()

if st.button(
    "🚘 Predict Price",
    type="primary",
    use_container_width=True
):

    st.session_state["car_details"] = {
        "Company_Name": company,
        "Brand": brand,
        "Model": int(model_year),
        "Condition": condition
    }

    st.switch_page("pages/2_Prediction.py")

