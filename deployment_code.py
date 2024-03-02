import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as f:
    model = pickle.load(f)



def forecast_next_three_months(column, order):
    try:
        model = ARIMA(column, order=order)
        res = model.fit()
        forecast_next_three_months = res.forecast(steps=3)
        return forecast_next_three_months
    except:
        return None




def main():
    st.title("Forecasted Values for Segregated Columns")

    # Display explanation
    st.write("This app provides forecasted values for segregated columns with MAPE less than or equal to 30.")

    # Load the predicted values for segregated columns
    predicted_df = pd.read_csv("predicted_values_next_three_months_segregated.csv")

    # Get the columns with MAPE less than or equal to 30
    segregated_columns = predicted_df.columns

    # Dropdown to select a column
    selected_column = st.selectbox("Select a column:", segregated_columns)

    # Load best_params_df
    best_params_df = pd.read_csv("best_params_per_column.csv", index_col=0)

    if st.button("Get Forecast"):
        st.write(f"Forecasted values for {selected_column}:")
        order = eval(best_params_df.loc[selected_column, 'best_params']) # Converting string to tuple
        forecast_values = forecast_next_three_months(predicted_df[selected_column], order)
        if forecast_values is not None:
            st.write(forecast_values)
        else:
            st.write("Error: Forecasting failed.")

if __name__ == "__main__":
    main()
