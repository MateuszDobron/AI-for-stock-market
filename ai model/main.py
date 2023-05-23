# ---- IMPORTS ----
# import inline as inline
# import matplotlib as matplotlib
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
# from datetime import datetime
import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import connexion
# ------------------
class AI:
    def __init__(self):
        np.set_printoptions(suppress=True)
        days = 22
        # Fetching the data
        StockDataCsv = pd.read_csv('s&p/sp500_index.csv', skiprows=[0], header=None, names=['Date', 'Closing'])
        print(StockDataCsv.shape)
        StockDataCsv.head()

        FullData = StockDataCsv.to_numpy()

        indexing_arr = np.arange(days + 1)
        indexing_arr = np.tile(indexing_arr, FullData.shape[0] - days)
        indexing_arr = np.resize(indexing_arr, (FullData.shape[0] - days, days + 1))
        indexing_arr_sub = np.arange(FullData.shape[0] - days)
        indexing_arr_sub = np.repeat(indexing_arr_sub, days + 1)
        indexing_arr_sub = np.resize(indexing_arr_sub, (FullData.shape[0] - days, days + 1))
        indexing_arr = indexing_arr + indexing_arr_sub
        FullDataNew = FullData[indexing_arr[np.arange(FullData.shape[0] - days), :], 1]
        FullDataNew = np.swapaxes(FullDataNew, 0, 1)
        print(FullDataNew[:, 0])
        print(FullDataNew.shape)

        # Feature Scaling for fast training of neural networks
        sc = MinMaxScaler()
        print(FullDataNew.shape)
        DataScaler = sc.fit(FullDataNew)
        X = DataScaler.transform(FullDataNew)
        print(X[:, 0])
        X_scaled = np.copy(X)
        X = np.swapaxes(X, 0, 1)
        X = np.reshape(X, (FullData.shape[0] - days, days + 1, 1), order='C')

        Y = X[:, days, :]
        X = X[:, 0:days, :]
        # #
        X_train = X[0:2300, :, :]
        X_test = X[2301:, :, :]
        y_train = Y[0:2300, :]
        y_test = Y[2301:, :]

        # ############################################
        #
        print(X_train[0, :, :])
        print(y_train[0, :])
        # Printing the shape of training and testing
        print('\n#### Training Data shape ####')
        print(X_train.shape)
        print(y_train.shape)
        print('\n#### Testing Data shape ####')
        print(X_test.shape)
        print(y_test.shape)
        # Defining Input shapes for LSTM
        TimeSteps = X_train.shape[1]
        TotalFeatures = X_train.shape[2]
        print("Number of TimeSteps:", TimeSteps)
        print("Number of Features:", TotalFeatures)

        # # Initialising the RNN
        self.regressor = Sequential()

        # Adding the First input hidden layer and the LSTM layer
        # return_sequences = True, means the output of every time step to be shared with hidden next layer
        self.regressor.add(LSTM(units=15, activation='tanh', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

        # return_sequences = True, means the output of every time step to be shared with hidden next layer
        self.regressor.add(LSTM(units=10, activation='tanh', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

        # return_sequences = True, means the output of every time step to be shared with hidden next layer
        self.regressor.add(LSTM(units=10, activation='tanh', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

        # Adding the Second Third hidden layer and the LSTM layer
        self.regressor.add(LSTM(units=5, activation='tanh', return_sequences=False))

        # Adding the output layer
        self.regressor.add(Dense(units=1))

        # Compiling the RNN
        self.regressor.compile(optimizer='adam', loss='mean_squared_error')

        ##################################################

        # Measuring the time taken by the model to train
        StartTime = time.time()

        # Fitting the RNN to the Training set
        self.regressor.fit(X_train, y_train, batch_size=5, epochs=1)

        EndTime = time.time()
        print("## Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes ##')

        # prediction for some existing entries
        # predicted_Price = regressor.predict(X_test)
        # predicted_Price = np.swapaxes(predicted_Price, 0, 1)
        # X_scaled[days, 2301:] = predicted_Price
        # predicted_Price = DataScaler.inverse_transform(X_scaled)
        #
        # print('Predicted price: ', predicted_Price[days, 2301:])

if __name__ == "__main__":
    #API
    app = connexion.App(__name__, specification_dir="./")
    app.add_api("swagger.yml")
    app.run(host="0.0.0.0", port=8000, debug=False)
