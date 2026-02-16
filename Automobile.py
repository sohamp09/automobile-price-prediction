import streamlit as st
import pandas as pd
import joblib

# Load ML files
ml_model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Car Price Prediction")

make = st.selectbox("Make", ["Toyota", "Honda", "Ford", "BMW", "Audi"])
year = st.number_input("Year", min_value=1900, max_value=2024, step=1)
fuel_type = st.selectbox("Fuel Type", ["Gas", "Diesel"])
engine_type = st.selectbox("Engine Type", ["V6", "V8", "I4"])
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
horsepower = st.number_input("Horsepower", min_value=0)
number_of_doors = st.selectbox("Number of Doors", [2, 4])
curb_weight = st.number_input("Curb Weight (lbs)", min_value=0)
highway_mpg = st.number_input("Highway MPG", min_value=0)
year_of_manufacture = st.number_input("Year of Manufacture", min_value=1900, max_value=2024, step=1)

if st.button("Predict Price"):

    raw_input = pd.DataFrame({
        "make": [make],
        "year": [year],
        "fuel_type": [fuel_type],
        "engine_type": [engine_type],
        "engine_size": [engine_size],
        "horsepower": [horsepower],
        "number_of_doors": [number_of_doors],
        "curb_weight": [curb_weight],
        "highway_mpg": [highway_mpg],
        "Year_of_Manufacture": [year_of_manufacture]
    })

    input_df = raw_input.copy()

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaler_input = scaler.transform(input_df)
    prediction = ml_model.predict(scaler_input)[0]

    st.success(f"Predicted Car Price: ${prediction:.2f}")
