import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("alphabet.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)

# Split into labels and images
X = dataset.drop('label',axis = 1)
y = dataset['label']


# Map label to alphabets
alphabets_class = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
dataset['label'] = dataset['label'].map(alphabets_class)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

X_train = np.array(X_train).reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = np.array(X_test).reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Creating Our model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(y.unique()), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=2)

model.save("model-alphabets.h5")

