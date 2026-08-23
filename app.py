import streamlit as st
import requests

# =========================================================
# Configuration
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       Main Page
       ===================================================== */

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* =====================================================
       Header
       ===================================================== */

    .title {
        text-align: center;
        color: #111827;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 35px;
    }


    /* =====================================================
       Section Titles
       ===================================================== */

    .section-title {
        color: #111827;
        font-size: 21px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* =====================================================
       Prediction Card
       ===================================================== */

    .prediction-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 30px 35px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    }

    .prediction-label {
        color: #6b7280;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .prediction-price {
        color: #111827;
        font-size: 40px;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .prediction-subtitle {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 8px;
    }


    /* =====================================================
       Property Summary
       ===================================================== */

    .summary-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 15px;
    }

    .summary-title {
        color: #374151;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .summary-item {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .summary-value {
        color: #111827;
        font-weight: 600;
    }


    /* =====================================================
       Predict Button
       ===================================================== */

    div.stButton > button {

        width: 100%;

        height: 52px;

        border-radius: 10px;

        font-size: 17px;

        font-weight: 600;

        background-color: #2563eb;

        color: white;

        border: none;

        transition: all 0.2s ease;

    }

    div.stButton > button:hover {

        background-color: #1d4ed8;

        color: white;

        border: none;

    }


    /* =====================================================
       Selectbox / Number Input
       ===================================================== */

    div[data-baseweb="select"] > div {

        border-radius: 9px;

    }

    div[data-testid="stNumberInput"] input {

        border-radius: 9px;

    }


    /* =====================================================
       Checkbox
       ===================================================== */

    div[data-testid="stCheckbox"] {

        margin-bottom: 5px;

    }


    /* =====================================================
       Success Message
       ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 10px;

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="title">🏠 House Price Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Predict the estimated value of a property using machine learning"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# Check FastAPI Connection
# =========================================================

try:

    response = requests.get(
        f"{API_URL}/",
        timeout=5,
    )

    if response.status_code != 200:

        st.error("FastAPI server is not responding correctly.")

        st.stop()

except requests.exceptions.RequestException:

    st.error("❌ FastAPI server is not running.")

    st.info("Start your FastAPI server using:\n\n" "`uvicorn main:app --reload`")

    st.stop()


# =========================================================
# Get Cities
# =========================================================

try:

    city_response = requests.get(
        f"{API_URL}/cities",
        timeout=10,
    )

    if city_response.status_code == 200:

        cities = city_response.json().get(
            "cities",
            [],
        )

    else:

        st.error("❌ Could not load cities from FastAPI.")

        st.stop()

except requests.exceptions.RequestException:

    st.error("❌ Could not connect to the `/cities` endpoint.")

    st.stop()


if not cities:

    st.warning("⚠️ No cities found in the API.")

    st.stop()


# =========================================================
# Property Location
# =========================================================

st.markdown(
    '<div class="section-title">' "📍 Property Location" "</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)


# =========================================================
# City Dropdown
# =========================================================

with col1:

    city = st.selectbox(
        "Select City",
        options=["Select City"] + cities,
        index=0,
    )


# =========================================================
# Location Dropdown
# =========================================================

locations = []


if city != "Select City":

    try:

        location_response = requests.get(
            f"{API_URL}/locations/{city}",
            timeout=10,
        )

        if location_response.status_code == 200:

            locations = location_response.json().get(
                "locations",
                [],
            )

        else:

            st.error("❌ Could not load locations for the selected city.")

    except requests.exceptions.RequestException:

        st.error("❌ Could not connect to the locations endpoint.")


with col2:

    if city == "Select City":

        location = st.selectbox(
            "Select Location",
            ["Select a city first"],
        )

    elif not locations:

        location = st.selectbox(
            "Select Location",
            ["No locations available"],
        )

    else:

        location = st.selectbox(
            "Select Location",
            ["Select Location"] + locations,
        )


# =========================================================
# Property Details
# =========================================================

st.markdown(
    '<div class="section-title">' "🏡 Property Details" "</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=50.0,
    )


with col2:

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )


# =========================================================
# Amenities
# =========================================================

st.markdown(
    '<div class="section-title">' "✨ Amenities" "</div>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)


with col1:

    gymnasium = st.checkbox("🏋️ Gymnasium")

    swimming_pool = st.checkbox("🏊 Swimming Pool")

    clubhouse = st.checkbox("🏢 Clubhouse")


with col2:

    power_backup = st.checkbox("🔋 Power Backup")

    car_parking = st.checkbox("🚗 Car Parking")

    lift_available = st.checkbox("🛗 Lift Available")


with col3:

    wardrobe = st.checkbox("👔 Wardrobe")

    ac = st.checkbox("❄️ Air Conditioning")

    wifi = st.checkbox("📶 WiFi")


# =========================================================
# Prediction Button
# =========================================================

st.markdown("")

predict_button = st.button("🔮 Predict House Price")


# =========================================================
# Prediction
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if city == "Select City":

        st.warning("⚠️ Please select a city.")

        st.stop()

    if location in [
        "Select Location",
        "Select a city first",
        "No locations available",
    ]:

        st.warning("⚠️ Please select a location.")

        st.stop()

    # -----------------------------------------------------
    # Request Payload
    # -----------------------------------------------------

    payload = {
        "area": area,
        "bedrooms": bedrooms,
        "city": city,
        "location": location,
        "gymnasium": int(gymnasium),
        "swimming_pool": int(swimming_pool),
        "clubhouse": int(clubhouse),
        "power_backup": int(power_backup),
        "car_parking": int(car_parking),
        "lift_available": int(lift_available),
        "wardrobe": int(wardrobe),
        "ac": int(ac),
        "wifi": int(wifi),
    }

    # -----------------------------------------------------
    # Call FastAPI
    # -----------------------------------------------------

    try:

        with st.spinner("Predicting property value..."):

            prediction_response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        # -------------------------------------------------
        # Successful Prediction
        # -------------------------------------------------

        if prediction_response.status_code == 200:

            result = prediction_response.json()

            prediction = float(result["Predicted_price"])

            # ---------------------------------------------
            # Indian Price Formatting
            # ---------------------------------------------

            if prediction >= 10000000:

                price_text = f"₹ {prediction / 10000000:.2f} Crore"

            elif prediction >= 100000:

                price_text = f"₹ {prediction / 100000:.2f} Lakh"

            else:

                price_text = f"₹ {prediction:,.0f}"

            # ---------------------------------------------
            # Prediction Card
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-box">

                    <div class="prediction-label">
                        Estimated House Price
                    </div>

                    <div class="prediction-price">
                        {price_text}
                    </div>

                    <div class="prediction-subtitle">
                        Estimated value based on the
                        property details provided
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------------------------------------
            # Property Summary
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="summary-card">

                    <div class="summary-title">
                        Property Summary
                    </div>

                    <div class="summary-item">
                        City:
                        <span class="summary-value">
                            {city}
                        </span>
                    </div>

                    <div class="summary-item">
                        Location:
                        <span class="summary-value">
                            {location}
                        </span>
                    </div>

                    <div class="summary-item">
                        Area:
                        <span class="summary-value">
                            {area:,.0f} sq ft
                        </span>
                    </div>

                    <div class="summary-item">
                        Bedrooms:
                        <span class="summary-value">
                            {bedrooms}
                        </span>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # API Error
        # -------------------------------------------------

        else:

            st.error("❌ Prediction failed.")

            st.code(prediction_response.text)

    # -----------------------------------------------------
    # Connection Error
    # -----------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error("❌ Could not connect to FastAPI.")

    # -----------------------------------------------------
    # Timeout Error
    # -----------------------------------------------------

    except requests.exceptions.Timeout:

        st.error("⏳ FastAPI request timed out.")

    # -----------------------------------------------------
    # General Error
    # -----------------------------------------------------

    except Exception as e:

        st.error(f"❌ Unexpected error: {str(e)}")
