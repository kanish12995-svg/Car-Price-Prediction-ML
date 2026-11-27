
import streamlit as st

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at top left, #1e293b 0%, #0f172a 35%, #020617 100%);
    color: white;
}
</style>
""", unsafe_allow_html=True)

home = st.Page(
    
    "home.py",
    title="Home",
    icon="🏠",
    default=True
)

prediction = st.Page(
    "pages/2_Prediction.py",
    title="Prediction Result",
    icon="🚘"
)

pg = st.navigation([home, prediction])
pg.run()

