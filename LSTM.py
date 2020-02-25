"""
This script is the core of the Neural Network. Data are read, processed and then fed to the stacked LSTM model.
Then, once the neural network is trained, a plot showing the performance of the neural network over the test is shown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler

step = 3
train_test_split_ratio = 0.8
train_val_split_ratio = 0.8

# data preparation
dcsv = pd.read_csv('../data_grouped.csv', usecols=['pressure'])
data = dcsv['pressure'].values

training_size = int(len(data)*train_test_split_ratio)
test_size = data.size - training_size

# splitting dataset into training set and test set
data_train = data[:training_size]
data_test = data[training_size:]

val_size = training_size - int(len(data_train)*train_val_split_ratio)
training_size = int(len(data_train)*train_val_split_ratio)

# splitting training set into actual training set and validation set
data_val = data_train[training_size:]
data_train = data_train[:training_size]


# feature scaling: this will help lstm model to converge faster
sc = MinMaxScaler(feature_range=(0,1))
data_train = data_train.reshape(-1,1)
data_val = data_val.reshape(-1,1)
data_test = data_test.reshape(-1,1)
data_train_scaled = sc.fit_transform(data_train)

#write normalization parameter of the training set in a file, they will be used for run the network
norm_par_file = open("../normalization_parameter.txt", "w+")
norm_par_file.write("min: " + str(sc.data_min_) + "\n")
norm_par_file.write("max: " + str(sc.data_max_))
norm_par_file.close()

data_test_scaled = sc.transform(data_test)
data_val_scaled = sc.transform(data_val)

# taking the last step measurements to predict the step + 1 pressure value
x_train=[]
y_train=[]

for i in range(step,training_size):
    x_train.append(data_train_scaled[i-step:i,0])
    y_train.append(data_train_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0]))

# taking the last step measurements to predict the step + 1 pressure value
x_val=[]
y_val=[]

for i in range(step,val_size):
    x_val.append(data_train_scaled[i-step:i,0])
    y_val.append(data_train_scaled[i,0])

x_val, y_val = np.array(x_val), np.array(y_val)

x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
y_val = np.reshape(y_val, (y_val.shape[0]))

# building the network
regressor = tf.keras.models.Sequential()
regressor.add(tf.keras.layers.LSTM(units=30,return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(tf.keras.layers.Dropout(rate=0.2))
regressor.add(tf.keras.layers.LSTM(units=30,return_sequences=False))
regressor.add(tf.keras.layers.Dropout(rate=0.2))
regressor.add(tf.keras.layers.Dense(units=1))

# compiling the network
regressor.compile(optimizer='adam', loss='mean_squared_error')

log_dir_ = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir_, write_graph=False)

# fitting the network
regressor.fit(x=x_train, y=y_train, epochs = 30, batch_size=64, validation_data=(x_val,y_val), verbose=2, callbacks=[tensorboard])

regressor.save('lstm.h5')

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

plt.plot(y_test, linewidth=0.1, label="Test data")
plt.plot(predictions, linewidth=0.1, label="Predictions")
plt.xlabel("Samples")
plt.ylabel("Atmospheric pressure (hPa)")
plt.legend(loc='lower left')
plt.savefig("nn_performance.svg", format="svg")
plt.show()