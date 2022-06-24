import numpy as np
from numpy import array
from numpy import argmax
from numpy import array_equal
from numpy import split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import preprocessing

class Encoder(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    def call(self, input_sequence):
    	encoder_output, hidden, carry = self.lstm(input_sequence)
    	return encoder_output, hidden, carry

#Design Attention layer
class Attention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(Attention, self).__init__()
        # Concat score function
        self.w1 = Dense(rnn_size)
        self.w2 = Dense(rnn_size)
        self.v =  Dense(1)

    def call(self, decoder_hidden, encoder_output):
        # Concat score function: va (dot) tanh(Wa (dot) concat(decoder_output + encoder_output))
        # Decoder output must be broadcasted to encoder output's shape first
        decoder_hidden = tf.expand_dims(decoder_hidden,  1) #shape (batch size, max_len,hidden_dim)

        # Concat => Wa => va
        # (batch_size, max_len, 2 * rnn_size) => (batch_size, max_len, rnn_size) => (batch_size, max_len, 1)
        score = self.v(tf.keras.activations.tanh(self.w1(decoder_hidden) + self.w2(encoder_output))) # score now has the shape of (batch_size, max len, 1)

        # alignment alignment_weights = softmax(score)
        alignment = tf.keras.activations.softmax(score, axis=-1) #(batch_size, max_len,1)

        
        # context vector c_t is the weighted average sum of encoder output
        context = alignment*encoder_output # (batch_size, hidden_dim,1)
        context = tf.reduce_sum(context, axis=1)

        return context, alignment

class Decoder(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_dim)
        self.hidden_dim = hidden_dim
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_state=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    def call(self, input_sequence, state):
        decoder_outputs, state_h, state_c = self.lstm(input_sequence, initial_state=state)

        return decoder_outputs, state_h, state_c

#Split a multivariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:22111, 1:], data[22111:, 1:]
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
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, :])
		# move along one time step
		in_start += 1
	return array(X),array(y)
 #train the model
def build_model(train, n_input):
	# prepare data
    train_x, train_y = to_supervised(train, n_input)
    train_x=np.asarray(train_x).astype(np.float32)
    train_y=np.asarray(train_y).astype(np.float32)
	# define parameters
    verbose, epochs, batch_size = 1, 10, 1
    n_timesteps_in, n_features, n_timesteps_out = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	# Setup the encoder layer
    encoder_inputs = tf.keras.Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
    encoder = Encoder(100)
    encoder_outputs, encoder_state_h, encoder_state_c = encoder(encoder_inputs)
    encoder_states = [encoder_state_h, encoder_state_c]

	#Setup Attention layer
    attention = Attention(100)
	#Setup the decoder layer
    decoder = Decoder(100)
    decoder_dense = Dense(n_features, activation='softmax')

	#Store the decoder'sequence output into an array
    all_outputs = []

	# 1 initial decoder's input data
    # Prepare initial decoder input data that just contains the start character 
    # Note that we made it a constant one-hot-encoded in the model
    # that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
    # one-hot encoded zero(0) is the start symbol
    inputs = tf.ones((batch_size, 1, n_features))
    #inputs[:,0,0] = 1 

    #initial decoder's state
    decoder_outputs = encoder_state_h
    states = encoder_states

    #decoder will only process one time step at a time.
    for _ in range(n_timesteps_out):
    	# 3 pay attention
    	# create the context vector by applying attention to
    	#deocder_outputs (last hidden state) + encoder_outputs (all hidden states)
    	context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)

    	context_vector = tf.expand_dims(context_vector, 1)

    	# 4 concatenate the input + context vector to find the next decoder's input
    	inputs = tf.concat([context_vector, inputs], axis = -1)

    	# 5 passing the concatenated vector to the LSTM
    	decoder_outputs, state_h, state_c =decoder(inputs, states)
    	outputs = decoder_dense(decoder_outputs)

    	# 6 Use the last hidden state for prediction the output
    	# save the current prediction
    	# we will concatenate all predictions later
    	outputs = tf.expand_dims(outputs, 1)
    	all_outputs.append(outputs)

    	# 7 Reassign the output as inputs for the next iteration
    	inputs = outputs
    	states = [state_h, state_c]
    # 8 we had cretaed a prediction list for the output sequence
    # convert the list to output array by concatenating all predictions
    # shape [batch_size, timesteps, features]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    #Define and compile the model
    model = tf.keras.models.Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adagrad', loss='mse', metrics=['accuracy'])
    # fit network
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 0)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2, callbacks =[es])
    model.save_weights('model2_att_5.h5')

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	# forecast the next week
	yhat = model.predict(data, verbose=1)
	return yhat  

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = list()
	# walk-forward validation over each week
	predictions = list()
	in_start = 0
	# step over the entire history one time step at a time
	for i in range(len(test)):
		# define the end of the input sequence
		in_end = in_start + n_input
		# ensure we have enough data for this instance
		if in_end <= len(test):
			history.append(test[in_start:in_end, :])
			# move along one time step
			in_start += 1
			# predict the week
			yhat_sequence = forecast(model, history, n_input)
			# store the predictions
			predictions.append(yhat_sequence)
			# clear this observation after predicting to store the next observation
			history.clear()
		else:
			break 
	# evaluate predictions days for each week
	predictions = array(predictions)
	#score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return predictions

# load the new file
dataset = read_csv('GWA-T-13_Materna-Workload-Traces/Materna-Trace-1/01.csv', sep=';', engine='python')
# split into train and test
train, test = split_dataset(dataset.values)
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.fit_transform(test)
# evaluate model and get scores
n_input = 50
build_model(train, n_input)
# visualize the loss training and validation
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Val'], loc='upper right')
#plt.show()

#visualize the training accuracy and the validation accuracy to see if the model is overfitting
#plt.plot(hist.history['accuracy'])
#plt.plot(hist.history['val_accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Val'], loc='lower right')
#plt.show()



