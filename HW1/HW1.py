import numpy as np
import pandas as pd

#weight = np.random.normal(size = 16*9)
weight = np.zeros(16*9)
biases = 0
data = np.empty([ 15*240, 16*9+1])
counter = 0

#import data and package into an object
for row in pd.read_csv( 'train.csv', na_values = "NR", keep_default_na = False, iterator = True, chunksize = 18):
    row = row.drop(row.index[[14,15]])
    row = row.set_index('num', inplace = False, drop = True)
    row = row.T
    for i in range(0,15):
        mtx = np.array(row[i:i+9])
        mtx = mtx.ravel()
        mtx = np.append(mtx,row.iloc[i+9,9])
        data[counter] = mtx
        counter = counter + 1

m = 100
diff_sum = np.zeros(16*9)
learn_rate = 0.1

while m:
    diff = np.empty([ 16*9])
    diffb = 0
    data_dot = np.empty([15*240])
    for i in range(0,15*240):
        datatmp = data[i]
        data_dot[i] = np.dot(datatmp[:16*9], weight)

    for k in range(0,16*9):
        num = 0
        for i in range(0,15*240):
            datatmp = data[i]
            num = num + ((-2) * (datatmp[16*9] - (biases + data_dot[i])) * (datatmp[k]))
        diff[k] = num + 2 * 100 * weight[k]

    num = 0
    for i in range(0,15*240):
        datatmp = data[i]
        num = num + ( 2 * (datatmp[16*9] - (biases + data_dot[i])))
    diffb = num

    for i in range(0,16*9):
        diff_sum[i] = diff_sum[i] + diff[i]**2
        tmp = learn_rate/(diff_sum[i]**(0.5))
        #print(diff_sum[i])
        weight[i] = weight[i] - 0.0000000012 * diff[i]
    biases = biases - 0.00000000012 * diffb
    
    m = m - 1

    error_sum = 0
    for i in range(0,15*240):
        datatmp = data[i]
        error = data_dot[i] + biases - datatmp[16*9]
        error_sum = error_sum + abs(error)

    print(error_sum/(15*240))