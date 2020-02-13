"""
This script is used in order to perform data wrangling on the initial dataset, which has raw data
that needed to be cleaned up.

The "raw data" comes in a .txt file with the following format:

<n_i>   201903312310  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  1016.9
<n_i>   201903312320  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  1016.8
<n_i>   201903312330  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  <n_i>  1017.0
...

Where the <n_i> tag stands for "not interested" in the analysis I have performed.
"""

import re
from datetime import datetime

# This function was used to produce a dataset to train the network in a time range equal to the foreseen one (8h).
def getTimeBand(datetime):
    if 0 <= datetime.hour < 8: return 0
    if 8 <= datetime.hour < 16: return 1
    if 16 <= datetime.hour <= 23: return 2

# This function was used to produce a dataset to train the network in a time range shorter than the original one
# (with this new setting, the prediction is provided each 40 minutes instead of each 8 hours). This has been done for
# testing purposes.
def getTimeBand40min(datetime):
    j=0
    for i in range(0, 23, 2):
        if datetime.hour == i and datetime.minute < 40: return j
        j+=1
        if datetime.hour == i and datetime.minute >= 40 or datetime.hour == i+1 and datetime.minute < 20: return j
        j+=1
        if datetime.hour == i+1 and datetime.minute >= 20: return j
        j+=1


raw_data_input = open("../raw_test_data.txt", "r")
data_output = open("../test_data.csv", "w+")
data_output.write("timeband,pressure\n")

datetime_index = 1;
pressure_measurement_index = 11

first_line = raw_data_input.readline()
first_line_splitted = re.split('  +', first_line)
avg_pressure_value = float(first_line_splitted[pressure_measurement_index])
n_values = 1
avg_datetime = datetime.strptime(first_line_splitted[datetime_index],"%Y%m%d%H%M")
for line in raw_data_input:
    line_splitted = re.split('  +', line)
    datetime = datetime.strptime(line_splitted[datetime_index],"%Y%m%d%H%M")
    #sometimes, pressure measurements are corrupted and they are not number. In those cases the measurement is simply skipped
    try:
        pressure_measurement = float(line_splitted[pressure_measurement_index])
    except ValueError:
        print("The value measured at time " + datetime + " cannot be parsed.")
        continue

    #check whether the pressure_measurement belongs to the same time band and same day of the avg_pressure_value,
    #if so update the avg, otherwise write in the output file and reinitialize the avg variables
    if datetime.date() == avg_datetime.date() and getTimeBand(datetime) == getTimeBand(avg_datetime):
        avg_pressure_value = avg_pressure_value + (pressure_measurement - avg_pressure_value) / n_values
        n_values += 1
    else:
        data_output.write(str(getTimeBand(avg_datetime)) + ',' + str(avg_pressure_value) + '\n')
        avg_pressure_value = pressure_measurement
        avg_datetime = datetime
        n_values = 1

data_output.close()




