
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional: if you have xgboost installed and want to ensure import for model loading
try:
    import xgboost  # noqa: F401
except Exception:
    pass

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="CodeX Beverage: Price Prediction", page_icon=":cup_with_straw:", layout="wide")

st.markdown("<h1 style='text-align:center'>CodeX Beverage: Price Prediction</h1>", unsafe_allow_html=True)
st.caption("XGBoost-powered price suggestion based on customer profile & preferences.")

# ---------- OPTIONS (edit as needed) ---------- #
GENDER_OPTS = ["M", "F", "Other"]
ZONE_OPTS = ["Urban", "Semi-Urban", "Rural"]
OCCUPATION_OPTS = ["Student", "Working Professional", "Homemaker", "Self-Employed", "Retired", "Other"]
INCOME_OPTS = ["<10L", "10-20L", "20-40L", "40-60L", "60-80L", "80L+"]
FREQ_OPTS = ["0-2 times", "3-5 times", "6-9 times", "10+ times"]
BRAND_OPTS = ["Newcomer", "Local", "Regional", "National", "International"]
SIZE_OPTS = ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)", "Family (2 L)"]
AWARENESS_OPTS = ["0 to 1", "2 to 3", "4 to 6", "7+"]
REASON_OPTS = ["Price", "Taste", "Brand", "Availability", "Quality", "Health"]
FLAVOR_OPTS = ["Traditional", "Fruity", "Mint", "Spicy", "Diet/Light"]
CHANNEL_OPTS = ["Online", "Kirana/Local Store", "Supermarket", "Vending", "Cafeteria"]
PACKAGING_OPTS = ["Simple", "Premium", "Eco-friendly", "Bulk"]
HEALTH_OPTS = ["Low (Not very concerned)", "Medium (Sometimes concerned)", "High (Very concerned)"]
SITUATION_OPTS = ["Active (eg. Sports, gym)", "Casual/At home", "Work/On the go", "Social/Party"]

DEFAULT_MIN_PRICE = 15.0
DEFAULT_MAX_PRICE = 75.0

# ---------- MODEL LOADING ---------- #
MODEL_FILE = Path("xgb_pipeline.pkl")  # Recommended: a sklearn Pipeline with preprocessing + XGBRegressor
MODEL_COLS_FILE = Path("model_columns.json")  # Optional: list of training columns for one-hot alignment

model = None
model_columns = None
if MODEL_FILE.exists() and (joblib is not None):
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        st.warning(f"Could not load model from {MODEL_FILE}: {e}")
if MODEL_COLS_FILE.exists():
    try:
        import json as _json
        model_columns = _json.loads(MODEL_COLS_FILE.read_text())
    except Exception as e:
        st.warning(f"Could not read model columns from {MODEL_COLS_FILE}: {e}")

def _fallback_predict(df: pd.DataFrame) -> np.ndarray:
    """A simple heuristic fallback so the app runs even without a model file."""
    base = 35.0
    # Nudge by income & size for a more realistic baseline
    income_factor = INCOME_OPTS.index(df.loc[0, "income_level"]) / max(1, len(INCOME_OPTS)-1)
    size_factor = SIZE_OPTS.index(df.loc[0, "preferable_size"]) / max(1, len(SIZE_OPTS)-1)
    health_factor = HEALTH_OPTS.index(df.loc[0, "health_concerns"]) / max(1, len(HEALTH_OPTS)-1)
    price = base + 20*income_factor + 10*size_factor + 5*health_factor
    return np.array([price])

def _predict(df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return _fallback_predict(df)

    # If the loaded object is a pipeline, it can usually accept raw df
    try:
        return model.predict(df)
    except Exception:
        # If it fails, try one-hot aligning to saved training columns (if provided)
        X = pd.get_dummies(df, drop_first=False)
        if model_columns is not None:
            # reindex to training columns (fill missing with 0)
            X = X.reindex(columns=model_columns, fill_value=0)
        return model.predict(X)

# ---------- UI LAYOUT ---------- #
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=90, step=1, value=28, help="Enter customer's age")
    income_level = st.selectbox("Income Level (In L)", INCOME_OPTS, index=0, key="income")
    awareness = st.selectbox("Awareness of other brands", AWARENESS_OPTS, index=0)
    packaging_pref = st.selectbox("Packaging Preference", PACKAGING_OPTS, index=0)

with col2:
    gender = st.selectbox("Gender", GENDER_OPTS, index=0)
    consume_freq = st.selectbox("Consume Frequency (weekly)", FREQ_OPTS, index=0)
    reason_choice = st.selectbox("Reasons for choosing brands", REASON_OPTS, index=0)
    health_concerns = st.selectbox("Health Concerns", HEALTH_OPTS, index=0)

with col3:
    zone = st.selectbox("Zone", ZONE_OPTS, index=0)
    occupation = st.selectbox("Occupation", OCCUPATION_OPTS, index=1)
    current_brand = st.selectbox("Current Brand", BRAND_OPTS, index=0)
    preferable_size = st.selectbox("Preferable Consumption Size", SIZE_OPTS, index=0)

col4, col5 = st.columns(2)
with col4:
    flavor_pref = st.selectbox("Flavor Preference", FLAVOR_OPTS, index=0)
with col5:
    purchase_channel = st.selectbox("Purchase Channel", CHANNEL_OPTS, index=0)

situation = st.selectbox("Typical Consumption Situations", SITUATION_OPTS, index=0)

# Collect into a single-row DataFrame to send to the model
input_df = pd.DataFrame([{
    "age": int(age),
    "gender": gender,
    "zone": zone,
    "occupation": occupation,
    "income_level": income_level,
    "consume_freq": consume_freq,
    "current_brand": current_brand,
    "preferable_size": preferable_size,
    "awareness_other_brands": awareness,
    "reason_choice": reason_choice,
    "flavor_pref": flavor_pref,
    "purchase_channel": purchase_channel,
    "packaging_pref": packaging_pref,
    "health_concerns": health_concerns,
    "situation": situation,
}])

st.markdown("---")
c1, c2 = st.columns([1,3])
with c1:
    go = st.button("Calculate Price Range", use_container_width=True)
with c2:
    st.write("")

if go:
    pred = float(_predict(input_df)[0])
    # Clamp to a sane range
    pred = float(np.clip(pred, DEFAULT_MIN_PRICE, DEFAULT_MAX_PRICE))
    # Show a +/- 10% band as a "range"
    low = round(pred * 0.9, 2)
    high = round(pred * 1.1, 2)

    st.success(f"**Suggested Price:** ₹{pred:.2f}")
    st.info(f"**Recommended Range:** ₹{low:.2f} — ₹{high:.2f}")

    with st.expander("See the features used"):
        st.dataframe(input_df)

# ---------- HOW TO CONNECT YOUR TRAINED MODEL ---------- #
with st.expander("How to plug in your trained XGBoost model"):
    st.markdown(
        "1. Train a scikit-learn **Pipeline** that includes preprocessing (e.g., OneHotEncoder) and ends with an **XGBRegressor**.\n"
        "2. Save it as `xgb_pipeline.pkl` in the same folder as this app (recommended). For example: `joblib.dump(pipeline, 'xgb_pipeline.pkl')`.\n"
        "3. (Optional) If you are not using a full pipeline, save the training columns after one-hot encoding to `model_columns.json` so the app can align columns.\n"
        "4. Run the app with: `streamlit run app.py`."
    )
