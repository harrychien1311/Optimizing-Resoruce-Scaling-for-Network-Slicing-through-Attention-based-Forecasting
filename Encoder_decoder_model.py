from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn import preprocessing
import numpy
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping 
 
# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:50000,1: ], data[5011:,1:]
	# restructure into windows of weekly data
	#train = array(split(train, len(train)/7))
	#test = array(split(test, len(test)/7))
	return train, test
 
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
 
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
 
# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=5):
	# flatten data
	#data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	data = train
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			X.append(data[in_start:in_end,:])
			y.append(data[in_end:out_end, :])
		# move along one time step
		in_start += 1
	return array(X), array(y)
 
# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input, 10)
	train_x=numpy.asarray(train_x).astype(numpy.float32)
	train_y=numpy.asarray(train_y).astype(numpy.float32)

	# define parameters
	verbose, epochs, batch_size = 1, 50, 100
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(Bidirectional(LSTM(300, activation='relu', input_shape=(n_timesteps, n_features))))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(300, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(n_features)))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	# fit network
	es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 3)
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2, callbacks =[es])
	model.save('model_GAN_30-10.h5')
 
# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
 
# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
 
# load the new file
col_list = ["CPU usage [%]", "Memory usage [%]", "Network received throughput [KB/s]"]
dataset = read_csv('GWA-T-13_Materna-Workload-Traces/Materna-Trace-1/01.csv',  sep=';', engine='python')
# split into train and test
train, test = split_dataset(dataset.values)
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.fit_transform(test)
# evaluate model and get scores
n_input = 30
build_model(train, n_input)
# summarize scores
#summarize_scores('lstm', score, scores)
# plot scores
#days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
#pyplot.plot(days, scores, marker='o', label='lstm')
#pyplot.show()