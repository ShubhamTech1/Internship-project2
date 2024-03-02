


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
#pip install keras
#pip install tensorflow

#pip install --upgrade keras
#pip install --upgrade tensorflow

#pip cache purge
#pip list --outdated


from keras.src.engine.base_layer import Layer

# Load the dataset
DF = pd.read_csv(r"C:\Users\Asus\clean_data.csv")  
DF.set_index('Dates', inplace = True)

# Data preprocessing
data = DF.iloc[:, 1:].values  # Assuming the first column is the timestamp
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Function to prepare data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Define number of time steps
n_steps = 3  # Change as needed

# Split into train and test sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Prepare data
X_train, y_train = prepare_data(train, n_steps)
X_test, y_test = prepare_data(test, n_steps)
print(X_train.shape)

# Reshape input data for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 88))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 88))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 88)))
model.add(Dense(88))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=88)

# Make predictions
y_pred = model.predict(X_test)

# Inverse scaling for predictions
y_pred = scaler.inverse_transform(y_pred)

# Inverse scaling for actual values
y_test_actual = scaler.inverse_transform(y_test)
# Save predictions and actual values to DataFrame with column names
results_df = pd.DataFrame({'Actual Values': y_test_actual.flatten(), 'Predicted Values': y_pred.flatten()})
results_df.columns = ['Actual_' + str(i) for i in range(1, len(y_test_actual[0]) + 1)] + ['Predicted_' + str(i) for i in range(1, len(y_pred[0]) + 1)]

# Save results to CSV with column names
results_df.to_csv("lstm_predictions.csv", index=False)

print("Results saved to lstm_predictions.csv.")

