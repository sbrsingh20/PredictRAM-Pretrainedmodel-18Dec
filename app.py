import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Streamlit App Title
st.title("Stock Return Predictor")

# Step 1: Upload Model File (.pkl)
uploaded_file = st.file_uploader("Upload your trained model file (.pkl):", type=["pkl"])

if uploaded_file is not None:
    try:
        # Load the model
        model = pickle.load(uploaded_file)
        st.success("Model loaded successfully!")
        st.write("Model Details:")
        st.write(model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.warning("Please upload a valid .pkl file to proceed.")

# Proceed only if the model is successfully loaded
if uploaded_file is not None and model is not None:
    # Step 2: User selects stocks
    st.header("Stock Selection")
    stock_options = ["Stock A", "Stock B", "Stock C", "Stock D", "Stock E"]
    selected_stocks = st.multiselect("Select stocks to predict returns for:", stock_options)

    # Step 3: User inputs for macroeconomic indicators
    st.header("Input Macroeconomic Indicators")
    gdp = st.number_input("GDP Growth Rate (%):", value=3.5, format="%.2f")
    inflation = st.number_input("Inflation Rate (%):", value=2.0, format="%.2f")
    interest_rate = st.number_input("Interest Rate (%):", value=1.5, format="%.2f")
    vix = st.number_input("VIX (Volatility Index):", value=20.0, format="%.2f")

    # Step 4: Predict button
    if st.button("Predict Stock Returns"):
        # Ensure stocks are selected
        if not selected_stocks:
            st.warning("Please select at least one stock to predict.")
        else:
            try:
                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    "GDP": [gdp],
                    "Inflation": [inflation],
                    "Interest_Rate": [interest_rate],
                    "VIX": [vix]
                })

                st.write("Input Data:")
                st.write(input_data)

                # Predict for each selected stock
                st.header("Predicted Stock Returns")
                for stock in selected_stocks:
                    # Here, assuming the model works directly with the input_data format
                    predicted_return = model.predict(input_data)[0]  # Adjust index if model outputs more
                    st.write(f"{stock}: Predicted Return: {predicted_return:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
