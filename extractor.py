import pandas as pd
from pandas import DataFrame
from pandas import concat
#from datetime import datetime

#def parse(x):
#  return datetime.strptime(x, '%Y %m %d')
#dataset = pd.read_csv('newdata.csv',  parse_dates = [['YEAR', 'MO', 'DY',]], index_col=0, date_parser=parse)


#getting the dataset
dataset = pd.read_csv('data.csv', header=0, index_col=0)
values = dataset.values

#plotting 
'''groups = [0, 1, 2, 3, 4, 5]

i = 1

from  matplotlib import pyplot

pyplot.figure()

for group in groups:
  pyplot.subplot(len(groups), 1, i)
  pyplot.plot(values[:, group])
  pyplot.title(dataset.columns[group], y=0.5, loc='right')
  i += 1
pyplot.show()'''


#function to convert series to supervised
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)
    return agg


#data preprocessing by normalizing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

#converting data to supervised
reframed = series_to_supervised(scaled, 1, 1)
#reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13, 14]], axis=1, inplace=True)
reframed.drop(reframed.columns[[6, 7, 8, 9, 10]], axis=1, inplace=True)


#splitting, 3 years of training and 2 years for testing
values = reframed.values
n_train_hours = 13152
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]


#doing this so nan value is replaced in training dataset
#traidpd = DataFrame(train)
#traidpd.fillna(traidpd.mean())
#train = traidpd.values

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.optimizers import SGD

model = Sequential()

model.add(LSTM(514, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(LeakyReLU(alpha=0.8))
optimizer = SGD(lr=0.001, decay=0.0001, momentum=0.9, nesterov=True)
model.compile(loss='mae', optimizer=optimizer)

# fit network
history = model.fit(train_X, train_y, epochs=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
from  matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
from numpy import concatenate
inv_yhat = concatenate((test_X[:, :-1], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:13,-1]

# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :-1], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:13,-1]

#error
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(inv_yhat, label='predicted')
pyplot.plot(inv_y, label='test')
pyplot.show()

