import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title of the Streamlit app
st.title("Stock Return Prediction App")

# File uploader for the .pkl file
uploaded_file = st.file_uploader("Upload the PKL file containing stock models", type="pkl")

if uploaded_file:
    # Load the .pkl file
    try:
        all_models = joblib.load(uploaded_file)
        st.success("PKL file successfully loaded!")

        # Display the content of the .pkl file
        st.subheader("Available Stocks in the Model")
        stocks = list(all_models.keys())
        st.write(stocks)

        # Allow user to select multiple stocks
        selected_stocks = st.multiselect("Select Stocks for Prediction", stocks)

        if selected_stocks:
            # Input fields for GDP, Inflation, Interest Rate, and VIX
            st.subheader("Enter Economic Indicators")
            gdp = st.number_input("GDP:", value=0.0, step=0.1)
            inflation = st.number_input("Inflation:", value=0.0, step=0.1)
            interest_rate = st.number_input("Interest Rate:", value=0.0, step=0.1)
            vix = st.number_input("VIX:", value=0.0, step=0.1)

            if st.button("Predict Stock Returns"):
                # Prepare the input data for prediction
                input_data = pd.DataFrame({
                    'GDP': [gdp],
                    'Inflation': [inflation],
                    'Interest Rate': [interest_rate],
                    'VIX': [vix]
                })

                # Predict returns for each selected stock
                predictions = {}
                for stock in selected_stocks:
                    try:
                        model = all_models[stock]['model']
                        predicted_return = model.predict(input_data)[0]
                        predictions[stock] = predicted_return
                    except Exception as e:
                        predictions[stock] = f"Error: {e}"

                # Display the predictions
                st.subheader("Predicted Stock Returns")
                for stock, return_value in predictions.items():
                    st.write(f"{stock}: {return_value:.4%}" if isinstance(return_value, (int, float)) else f"{stock}: {return_value}")
    except Exception as e:
        st.error(f"Error loading PKL file: {e}")
else:
    st.info("Please upload the PKL file to proceed.")
