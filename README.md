# Atm pressure predictor
This neural network is part of a bigger project in which the `lstm.h5` file is converted in a STM32-optimized library through the X-CUBE-AI expansion pack for STM32CubeMX. This library will be included and used in a NUCLEO-F401RE which will acquire atmospheric pressure data and use the STM32-optimized library to provide predictions over those data.

## The neural network
It is a stacked LSTM model which takes as input three pressure values and outputs the predicted next pressure value. The pressure values which this LSTM model takes as input must be grouped in time bands, so a single pressure value corresponds to the average of the pressure measurements acquired in the following time bands:

- Time band 1 (pressure measurements gathered from 00:00 to 07:59)
- Time band 2 (pressure measurements gathered from 08:00 to 15:59)
- Time band 3 (pressure measurements gathered from 16:00 to 23:59)

Given the pressure values of the last three time bands the LSTM is able to predict the pressure value of the next time band.

## Performance on the test set
The testing over a test set has performed as follows

<img src="https://user-images.githubusercontent.com/19633559/72997782-1796cb00-3dfd-11ea-981b-4a00d616c1d9.jpeg" width="600">

Where the orange line represents the predictions and the blue line represents the actual test value.


## Usage
The `requirements.txt` contains all the packages used for developing this neural network. Type `pip install -r requirements.txt` to install them all.
