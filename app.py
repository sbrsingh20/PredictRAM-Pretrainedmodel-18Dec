import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Streamlit App Title
st.title("Stock Return Prediction Application")

# Step 1: Upload Model File (.pkl)
uploaded_file = st.file_uploader("Upload your trained model file (.pkl):", type=["pkl"])

if uploaded_file is not None:
    try:
        # Load the model or data from the .pkl file
        model_data = pickle.load(uploaded_file)

        # Display the contents of the loaded file to check its structure
        st.success("File loaded successfully!")
        
        # Check if the model_data is a dictionary with stock file names
        if isinstance(model_data, dict):
            stock_names = list(model_data.keys())  # Stock names are the keys in the dictionary
            st.write("Available Stocks in the Model:")
            st.write(stock_names)

            # Create a dictionary for models and evaluation metrics
            stock_models = {stock: model_data[stock]['model'] for stock in stock_names}
            stock_evaluations = {stock: model_data[stock]['evaluation'] for stock in stock_names}
            
            # Show evaluation metrics for each stock
            st.write("Model Evaluation Metrics:")
            for stock, evaluation in stock_evaluations.items():
                st.write(f"**{stock}:**")
                st.write(f"  - Model Type: {evaluation['model_type']}")
                st.write(f"  - R2 Score: {evaluation['r2_score']}")
                st.write(f"  - Mean Squared Error: {evaluation['mean_squared_error']}")
                st.write(f"  - Accuracy: {evaluation['accuracy']}")
                st.write("---")
        else:
            st.warning("The uploaded file does not contain the expected dictionary structure.")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("Please upload a valid .pkl file to proceed.")

# Step 2: Prediction Flow
if 'stock_models' in locals():
    # Step 3: User selects stocks
    st.header("Select Stocks for Prediction")
    selected_stocks = st.multiselect("Select stocks to predict returns for:", stock_names)

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
                    # Check if the model is callable and make predictions
                    model = stock_models[stock]
                    if hasattr(model, 'predict'):
                        predicted_return = model.predict(input_data)[0]  # Adjust index if model outputs more
                        st.write(f"{stock}: Predicted Return: {predicted_return:.2f}%")
                    else:
                        st.warning(f"Model for {stock} does not have a predict method.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No model is available to make predictions. Please upload a valid .pkl file containing a trained model.")
