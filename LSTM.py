import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# data preparation
dcsv = pd.read_csv('../data_grouped.csv', usecols=['pressure'])
data = dcsv['pressure'].values

# feature scaling: this will help lstm model to converge faster
sc = MinMaxScaler(feature_range=(0,1))
data = data.reshape(-1,1)
data_scaled = sc.fit_transform(data)

# splitting dataset into training set and test set
data_train = data_scaled[:18000]
data_test = data_scaled[18000:]

# taking the last 30 measurement to predict the 31st pressure value
x_train=[]
y_train=[]

for i in range(30,18000):
    x_train.append(data_train[i-30:i,0])
    y_train.append(data_train[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0]))

# building the network
regressor = Sequential()
regressor.add(CuDNNLSTM(units=30,return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
regressor.add(CuDNNLSTM(units=30,return_sequences=False))
regressor.add(Dropout(rate=0.2))
regressor.add(Dense(units=1))

# compiling the network
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting the network
regressor.fit(x=x_train, y=y_train, epochs = 45, batch_size=64, verbose=2)

regressor.save('lstm.h5')

y_test = []
predictions = []

for i in range(30, 1939):
    x_test = []
    x_test.append(data_test[i-30:i,0])
    y_test.append(sc.inverse_transform(np.array(data_test[i]).reshape(1,-1)))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (1,30,1))
    predictions.append(sc.inverse_transform(np.array(regressor.predict(x=x_test)).reshape(1,-1)))

y_test = np.array(y_test)
predictions = np.array(predictions)
y_test = y_test.squeeze()
predictions = predictions.squeeze()

plt.plot(y_test)
plt.plot(predictions)
plt.savefig("plot.jpeg", dpi=1200)
plt.show()

