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

def prepare_input_dataset(data, data_size):
    # taking the last step measurements to predict the step + 1 pressure value
    x = []
    y = []

    for i in range(step, data_size):
        x.append(data[i - step:i, 0])
        y.append(data[i, 0])

    x, y = np.array(x), np.array(y)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0]))
    return x, y

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

x_train, y_train = prepare_input_dataset(data_train_scaled, training_size)

x_val, y_val = prepare_input_dataset(data_val_scaled, val_size)


# building the network
regressor = tf.keras.models.Sequential()
regressor.add(tf.keras.layers.LSTM(units=50,return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(tf.keras.layers.Dropout(rate=0.2))
regressor.add(tf.keras.layers.LSTM(units=50,return_sequences=False))
regressor.add(tf.keras.layers.Dropout(rate=0.2))
regressor.add(tf.keras.layers.Dense(units=1))

# compiling the network
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

log_dir_ = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir_, write_graph=False)

# fitting the network
regressor.fit(x=x_train, y=y_train, epochs = 100, batch_size=64, validation_data=(x_val,y_val), verbose=2, callbacks=[tensorboard])

regressor.save('lstm.h5')



# testing the model
x_test_ev, y_test_ev = prepare_input_dataset(data_test_scaled, test_size)
print(regressor.metrics_names, regressor.evaluate(x_test_ev, y_test_ev, batch_size=64))

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