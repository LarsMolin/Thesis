import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('preprocessed_data.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
df.set_index('Date', inplace=True)
df.index = df.index.tz_convert('Europe/Amsterdam')



X = df[[
        #'DAY AHEAD FORECAST (ENAPPSYS)',
        #'SOLAR FORECAST UNADJUSTED (ENAPPSYS)',
        #'WIND FORECAST UNADJUSTED (ENAPPSYS)',
        #'FORECAST D-1 (ENAPPSYS)',
        'NETTO SOLAR',
        'NETTO WIND',
        #'INFERRED INSTALLED CAPACITY (ENAPPSYS)_x',
        #'INFERRED INSTALLED CAPACITY (ENAPPSYS)_y',
        'DA Scheduled Flow',
        'Scheduled Flow', 
        'MinuteOfDay', 
        'HourOfDay',
        #'Dispatch -5m',
        #'Imbalance_Price -5m',
        #'IGCC_Dispatch -5m',
        #'Dispatch -2H',
        #'IGCC_Dispatch -2H',
        'Imbalance_Price -2H',
        #'SOLAR OUTTURN',
        #'WIND OUTTURN',
        #'minmax',
        'min600',
        'min300',
        'min100',
        'minmin',
        'posmin',
        'pos100',
        'pos300',
        'pos600',
        #'posmax',
        'NETHERLANDS (NL)'
        ]]
                    
        
y = df[['Final_Price']]#, 'Final_Price +2H']]#, 'Dispatch']]


# Assuming X and y are Pandas DataFrames
X = X.values  # Convert X to a NumPy array
y = y.values  # Convert y to a NumPy array

# Define the sequence length (number of time steps)
sequence_length = 60  # You can adjust this based on your data and problem

# Create sequences for LSTM
X_sequences = []
y_sequences = []
timestamps = []

for i in range(sequence_length, len(X)):
    X_sequences.append(X[i - sequence_length:i, :])
    y_sequences.append(y[i])
    timestamps.append(df.index[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
timestamps = np.array(timestamps).reshape((-1,1))
del(X,y,df)
# Define the size of each segment in minutes
training_segment = int(6 * 30.5 * 24 * 60)  # 6 months
validation_test_segment = int(30.5 * 24 * 60)  # 1 month

# Initialize variables to store MAE scores
mae_scores0 = []
mae_scores1 = []
rmse_scores0 = []
rmse_scores1 = []
month_counter = 0
it_counter = 0
early_stopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
for i in range(0, len(X_sequences) - training_segment, validation_test_segment):
    train_start = i
    train_end = i + training_segment
    val_start = train_end
    val_end = val_start + validation_test_segment
    test_start = val_end
    test_end = test_start + validation_test_segment

    X_train = X_sequences[train_start:train_end]
    y_train = y_sequences[train_start:train_end]
    X_val = X_sequences[val_start:val_end]
    y_val = y_sequences[val_start:val_end]
    X_test = X_sequences[test_start:test_end]
    y_test = y_sequences[test_start:test_end]

    X_scaler = MinMaxScaler()
    samples, time_steps, features = X_train.shape
    X_train_2d = X_train.reshape((-1, features))
    print(X_train_2d.shape)
    X_train_scaled_2d = X_scaler.fit_transform(X_train_2d)
    X_train = X_train_scaled_2d.reshape((samples, time_steps, features))

    samples, time_steps, features = X_val.shape
    X_val_2d = X_val.reshape((-1, features))
    X_val_scaled_2d = X_scaler.transform(X_val_2d)
    X_val = X_val_scaled_2d.reshape((samples, time_steps, features))

    samples, time_steps, features = X_test.shape
    X_test_2d = X_test.reshape((-1, features))
    X_test_scaled_2d = X_scaler.transform(X_test_2d)
    X_test = X_test_scaled_2d.reshape((samples, time_steps, features))

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    model = Sequential()
    model.add(LSTM(units=16, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    #model.add(Dropout(0.4))  # Add dropout here
    #model.add(LSTM(units=16, return_sequences=True))
    #model.add(Dropout(0.4))  # Add dropout here
    #model.add(LSTM(units=16))
    #model.add(Dropout(0.4))  # Add dropout here
    #model.add(Dense(units=16, activation='relu'))
    #model.add(Dropout(0.4))  # Add dropout here
    #model.add(Dense(units=16, activation='relu'))
    #model.add(Dropout(0.4))  # Add dropout here
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.2))  # Add dropout here
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=20, batch_size=8, callbacks=[early_stopping])

    y_test_prediction = model.predict(X_test)
    y_test_prediction = y_scaler.inverse_transform(y_test_prediction)
    y_test = y_scaler.inverse_transform(y_test)

    plot_range_start = 0
    plot_range_end = 500
    start_idx = test_start#i + test_start+plot_range_start
    end_idx = test_end #start_idx + plot_range_end
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps[start_idx:end_idx], y_test[:, 0], label='True Values')
    plt.plot(timestamps[start_idx:end_idx], y_test_prediction[:, 0], label='Predicted Values')
    plt.title("Prediction of 2 Hours Ahead")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.savefig(f'Predicted Price 2H Month{month_counter}.png', bbox_inches='tight')
    #plt.show()  # Display the plot

    # Plot for Column 1 (Volume)
    #plt.figure(figsize=(15, 6))
    #plt.plot(timestamps[start_idx:end_idx], y_test[:, 1], label='True Values')
    #plt.plot(timestamps[start_idx:end_idx], y_test_prediction[:, 1], label='Predicted Values')
    #plt.title("Prediction of 2 Hours Ahead")
    #plt.xlabel("Timestamp")
    #plt.ylabel("Value")
    #plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    #plt.legend()
    #plt.savefig(f'Predicted Price 2H Ahead Month{month_counter}.png')

    mae_0 = mean_absolute_error(y_test[:,0], y_test_prediction[:,0])
    rmse_0 = mean_squared_error(y_test[:,0], y_test_prediction[:,0], squared=False)
    #mae_1 = mean_absolute_error(y_test[:,1], y_test_prediction[:,1])
    #rmse_1 = mean_squared_error(y_test[:,1], y_test_prediction[:,1], squared=False)
    mae_scores0.append(mae_0)
    #mae_scores1.append(mae_1)
    rmse_scores0.append(rmse_0)
    #rmse_scores1.append(rmse_1)
    month_counter += 1
    it_counter += 1
    if it_counter > 3:
        break

# Calculate and print the average MAE across all segments
average_mae0 = np.mean(mae_scores0)
print(f'Average MAE across all months: {average_mae0}')
average_rmse0 = np.mean(rmse_scores0)
print(f'Average RMSE across all months: {average_rmse0}')
#average_mae1 = np.mean(mae_scores1)
#print(f'Average MAE across all months: {average_mae1}')
#average_rmse1 = np.mean(rmse_scores1)
#print(f'Average RMSE across all months: {average_rmse1}')

# You can also visualize the MAE scores for each segment if needed
for i, mae in enumerate(mae_scores0):
    print(f'Month {i + 1} MAE: {mae}')

for i, rmse in enumerate(rmse_scores0):
    print(f'Month {i + 1} RMSE: {rmse}')

#for i, mae in enumerate(mae_scores1):
#    print(f'Month {i + 1} MAE: {mae}')

#for i, rmse in enumerate(rmse_scores1):
#    print(f'Month {i + 1} RMSE: {rmse}')

with open('2H Ahead NMAE and RMSE.txt', mode='w') as file:
    file.write(f'Average MAE 5min ahead across all Months: {average_mae0}\n')
    for i,mae in enumerate(mae_scores0):
        file.write(f'Month {i + 1} MAE 5min ahead: {mae}\n')
    file.write(f'Average RMSE 5min ahead across all Months: {average_rmse0}')
    for i,rmse in enumerate(rmse_scores0):
        file.write(f'Month {i+1} RMSE 5min ahead: {rmse}\n')
    #file.write(f'Average MAE 2H ahead across all Months: {average_mae1}')
    #for i,mae in enumerate(mae_scores1):
    #    file.write(f'Month {i+1} MAE 2H ahead: {mae}\n')
    #file.write(f'Average RMSE 2H ahead across all Months: {average_rmse1}')
    #for i,rmse in enumerate(rmse_scores1):
    #    file.write(f'Month {i+1} RMSE 2H ahead: {rmse}\n')