# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 03:11:01 2018

@author: jaydeep thik
"""

import keras
from keras.datasets import imdb
from keras import preprocessing
from keras import models, layers

max_features = 1000
maxlen = 500

callbacks =[keras.callbacks.TensorBoard(log_dir='graphs', histogram_freq=1, embeddings_freq=1)]

(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

model  = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=maxlen, name='embed'))
model.add(layers.Conv1D(32, 7,activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
#model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,callbacks=callbacks)

