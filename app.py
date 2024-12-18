import streamlit as st
import pickle
import pandas as pd
import numpy as np

def load_model(file):
    """Load the uploaded pickle file."""
    try:
        model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_returns(model, selected_stocks, gdp, inflation, interest_rate, vix):
    """
    Predict stock returns based on user inputs and selected stocks.
    :param model: The pre-trained model
    :param selected_stocks: List of selected stocks
    :param gdp: GDP value
    :param inflation: Inflation value
    :param interest_rate: Interest Rate value
    :param vix: VIX value
    :return: DataFrame with predictions for selected stocks
    """
    try:
        # Create input data for prediction
        input_data = pd.DataFrame({
            'Stock': selected_stocks,
            'GDP': [gdp] * len(selected_stocks),
            'Inflation': [inflation] * len(selected_stocks),
            'Interest_Rate': [interest_rate] * len(selected_stocks),
            'VIX': [vix] * len(selected_stocks)
        })

        # Ensure the model supports prediction
        if not hasattr(model, "predict"):
            st.error("The uploaded model does not support prediction.")
            return None

        # Make predictions
        input_features = input_data.drop(columns=['Stock'])
        predictions = model.predict(input_features)

        # Add predictions to the DataFrame
        input_data['Predicted_Return'] = predictions
        return input_data

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit App
st.title("Stock Return Predictor")

# Step 1: Upload the pickle file
st.header("Step 1: Upload Model File (.pkl)")
uploaded_file = st.file_uploader("Upload your trained model (.pkl)", type="pkl")

if uploaded_file:
    model = load_model(uploaded_file)

    if model:
        st.success("Model loaded successfully!")
        st.write("Model Content (attributes and methods):")
        st.write(dir(model))

        # Step 2: Allow user to select stocks
        st.header("Step 2: Select Stocks")
        stock_list = [
            "AJANTPHARM", "AUBANK", "TCS", "INFY", "RELIANCE", "HDFCBANK", "ITC", "ONGC"
        ]  # Replace with dynamic stock names if needed
        selected_stocks = st.multiselect("Select stocks to predict returns:", stock_list)

        if selected_stocks:
            # Step 3: Input macroeconomic parameters
            st.header("Step 3: Input Macroeconomic Parameters")
            gdp = st.number_input("GDP (in %):", min_value=-100.0, max_value=100.0, value=2.5)
            inflation = st.number_input("Inflation (in %):", min_value=-100.0, max_value=100.0, value=5.0)
            interest_rate = st.number_input("Interest Rate (in %):", min_value=-100.0, max_value=100.0, value=4.0)
            vix = st.number_input("VIX (Volatility Index):", min_value=0.0, max_value=100.0, value=20.0)

            # Step 4: Predict and display results
            if st.button("Predict Stock Returns"):
                results = predict_returns(model, selected_stocks, gdp, inflation, interest_rate, vix)

                if results is not None:
                    st.header("Prediction Results")
                    st.dataframe(results)

                    # Option to download predictions
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predicted_stock_returns.csv",
                        mime="text/csv"
                    )
