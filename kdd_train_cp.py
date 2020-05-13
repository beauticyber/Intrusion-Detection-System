# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:55:44 2019

@author: Shoaib Hayat
"""

import h5py    
import numpy as np   
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, SpatialDropout1D, Conv1D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from pandas import Series
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from math import sqrt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



data = pd.read_csv('X_data.csv')
labels = pd.read_csv('Y_data.csv')

class_names =['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap' 'ipsweep', 'land' 'loadmodule', 'multihop,' 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']



#model = Sequential()

num_features = data.shape[0]
training_length = data.shape[1]
# Embedding layer

X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
#
le = LabelEncoder()
X_train_le = le.fit_transform(X_train)
X_test = le.fit_transform(X_test)

#
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

num_features = data.shape[0]
model = Sequential()

model.add(Embedding(num_features, 100, input_length=data.shape[1], trainable=False, mask_zero=True))
# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))
# Recurrent layer
model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
# Fully connected layer
model.add(Dense(64, activation='relu'))
# Dropout for regularization
model.add(Dropout(0.5))
# Output layer
model.add(Dense(23, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

epochs = 1
batch_size = 64
history = model.fit(X_train, Y_train,  epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test))  # starts training

accr = model.evaluate(X_test,Y_test)
prediction = model.predict(X_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}%'.format(accr[0],accr[1]*100))
#confusion_matrix(X_test, Y_test, True)
#results = confusion_matrix(expected, predicted)
#print(results)
# Loss Plot
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

# Accuracy Plot
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();

#
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
prediction = (prediction > 0.5).astype('int')
# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, prediction, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, prediction, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

