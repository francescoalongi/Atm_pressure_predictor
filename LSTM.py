import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


step = 3

# data preparation
dcsv = pd.read_csv('../data_grouped.csv', usecols=['pressure'])
data = dcsv['pressure'].values

# splitting dataset into training set and test set
data_train = data[:18000]
data_test = data[18000:]

# feature scaling: this will help lstm model to converge faster
sc = MinMaxScaler(feature_range=(0,1))
data_train = data_train.reshape(-1,1)
data_test = data_test.reshape(-1,1)
data_train_scaled = sc.fit_transform(data_train)
data_test_scaled = sc.transform(data_test)


# taking the last step measurements to predict the step + 1 pressure value
x_train=[]
y_train=[]

for i in range(step,18000):
    x_train.append(data_train_scaled[i-step:i,0])
    y_train.append(data_train_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0]))


# building the network
regressor = tf.keras.models.Sequential()
regressor.add(tf.keras.layers.LSTM(units=30,return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(tf.keras.layers.Dropout(rate=0.2))
regressor.add(tf.keras.layers.LSTM(units=30,return_sequences=False))
regressor.add(tf.keras.layers.Dropout(rate=0.2))
regressor.add(tf.keras.layers.Dense(units=1))

# compiling the network
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting the network
regressor.fit(x=x_train, y=y_train, epochs = 30, batch_size=64, verbose=2)

regressor.save('lstm.h5')


#regressor = tf.keras.models.load_model('../lstm.h5')
# testing the model
y_test = []
predictions = []
for i in range(step, 1939):
    x_test = []
    x_test.append(data_test_scaled[i-step:i,0])
    y_test.append(np.array(data_test[i]).reshape(1,-1))
    x_test = np.reshape(x_test, (1,step,1))
    predictions.append(sc.inverse_transform(np.array(regressor.predict(x=x_test)).reshape(1,-1)))

y_test = np.array(y_test)
predictions = np.array(predictions)
y_test = y_test.squeeze()
predictions = predictions.squeeze()

plt.plot(y_test)
plt.plot(predictions)
plt.savefig("plot.jpeg", dpi=1200)
plt.show()

