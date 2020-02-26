from pandas import read_csv
from pandas import DataFrame

dataset = read_csv('data.csv', header=0)

from matplotlib import pyplot

pyplot.figure()

y = (dataset.iloc[:,-1])

x= dataset.iloc[:,0]

pyplot.plot(x,y)

