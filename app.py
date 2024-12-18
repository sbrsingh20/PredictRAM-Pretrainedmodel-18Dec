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
        # Load the model or data from the .pkl file
        model_data = pickle.load(uploaded_file)
        
        # Check if the loaded object is a trained model
        if hasattr(model_data, 'predict'):
            model = model_data
            st.success("Model loaded successfully!")
            st.write("Model Details:")
            st.write(model)
        else:
            # If it's not a model, assume it's stock data or other data
            model = None
            st.warning("The file doesn't contain a model, displaying data instead.")
            st.write("Data in the file:")
            st.write(model_data)
            
            # If it's a DataFrame or similar, show the accuracy (if applicable)
            if isinstance(model_data, pd.DataFrame):
                st.write("Displaying stock data available in the .pkl file:")
                st.dataframe(model_data)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        model = None
else:
    st.warning("Please upload a valid .pkl file to proceed.")

# Step 2: Prediction Flow
if model is not None:
    # Step 3: User selects stocks
    st.header("Stock Selection")
    stock_options = ["Stock A", "Stock B", "Stock C", "Stock D", "Stock E"]
    selected_stocks = st.multiselect("Select stocks to predict returns for:", stock_options)

    # Step 4: User inputs for macroeconomic indicators
    st.header("Input Macroeconomic Indicators")
    gdp = st.number_input("GDP Growth Rate (%):", value=3.5, format="%.2f")
    inflation = st.number_input("Inflation Rate (%):", value=2.0, format="%.2f")
    interest_rate = st.number_input("Interest Rate (%):", value=1.5, format="%.2f")
    vix = st.number_input("VIX (Volatility Index):", value=20.0, format="%.2f")

    # Step 5: Predict button
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
                    # Check if the model is callable
                    if hasattr(model, 'predict'):
                        predicted_return = model.predict(input_data)[0]  # Adjust index if model outputs more
                        st.write(f"{stock}: Predicted Return: {predicted_return:.2f}%")
                    else:
                        st.warning("Model does not have a predict method.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No model is available to make predictions. Please upload a valid .pkl file containing a trained model.")
