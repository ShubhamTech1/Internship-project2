



import pandas as pd 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import pickle
from sqlalchemy import create_engine

user = 'root' # user name
pw = 'root' # password
db = 'project163' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# Load the dataset
DF = pd.read_csv(r"C:\Users\Asus\clean_data.csv")  

# dumping data into database 
DF.to_sql('KitItem', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# loading data from database
sql = 'select * from KitItem'

DF = pd.read_sql_query(sql, con = engine )
print(DF)




DF = pd.read_csv(r"C:\Users\Asus\clean_data.csv")  

# Data Partition
Train = DF.head(30)
Test = DF.tail(3)
 

# Define the range of p, d, and q values to try
# Define the range of p, d, and q values to try
p_values = range(3, 13)  # p values ranging from 3 to 12
d_values = [1]
q_values = range(3, 13)  # q values ranging from 3 to 12


# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   
best_params_per_column = {}  # Dictionary to store best parameters for each column

# Model Building and Forecasting for each column with hyperparameter tuning
for column in DF.columns[1:]:
    print(f"Forecasting for column: {column}")
    best_mape = np.inf
    best_params = None
    best_forecast = None
    for p, d, q in product(p_values, d_values, q_values):
        try:
            # ARIMA model with current hyperparameters
            model = ARIMA(Train[column], order=(p, d, q))
            res = model.fit()

            # Forecast for next 3 months
            forecast_test = res.forecast(steps=3)

            # Calculate MAPE
            mape = mean_absolute_percentage_error(Test[column], forecast_test)

            # Update best MAPE, parameters, and forecast if current MAPE is better
            if mape < best_mape:
                best_mape = mape
                best_params = (p, d, q)
                best_forecast = forecast_test
        except:
            continue
    
    # Store best parameters for the column
    best_params_per_column[column] = (best_params, best_mape)

# Convert best_params_per_column dictionary to DataFrame
best_params_df = pd.DataFrame.from_dict(best_params_per_column, orient='index', columns=['best_params', 'best_mape'])

# Save the DataFrame to CSV
best_params_df.to_csv("best_params_per_column.csv")

print("Best p, d, q values for each column saved to best_params_per_column.csv.")

# Segregate columns with MAPE <= 30
segregated_columns = [column for column, (params, mape) in best_params_per_column.items() if mape <= 30]

# Predict values for segregated columns using the best parameters
predicted_values_per_column = {}

for column, (params, mape) in best_params_per_column.items():
    if column in segregated_columns:
        try:
            model = ARIMA(DF[column], order=params)
            res = model.fit()
            forecast_next_three_months = res.forecast(steps=3)  # Predict next 3 months
            predicted_values_per_column[column] = forecast_next_three_months
        except:
            continue

# Create DataFrame for predicted values
predicted_df = pd.DataFrame(predicted_values_per_column)

# Save predicted values to CSV
predicted_df.to_csv("predicted_values_next_three_months_segregated.csv", index=False)

print("Predicted values for next three months for segregated columns saved to predicted_values_next_three_months_segregated.csv.")
