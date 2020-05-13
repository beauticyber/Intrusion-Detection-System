# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:55:44 2019

@author: Shoaib Hayat
"""
from numpy import array
from numpy import argmax

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, SpatialDropout1D, Conv1D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn import preprocessing

import pandas as pd
import numpy as np
import h5py



from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('data.csv' ,index_col=None)
class_names =['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap' 'ipsweep', 'land' 'loadmodule', 'multihop,' 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']

names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class_name','data']
#dataset = dataset.to_csv(index=False)
#Y = dataset.iloc[:,41:42].astype('U')
Y = dataset['class_name'].copy()
X = dataset.iloc[:,0:41]
#Y = np.array(Y)
#Y = np.concatenate(Y).astype('U')
classNames = np.unique(Y).astype('U')
values = array(Y)
#print(X.head())
#classNames = ['normal', 'neptune', 'warezclient', 'ipsweep', 'portsweep', 'teardrop', 'nmap', 'satan']
X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
print(Y)
Y = LabelBinarizer().fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 41)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)



#
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)




#

#le = LabelEncoder()
#X_train_le = le.fit_transform(X_train)
#X_test = le.fit_transform(X_test)
#
##values = array(Y)
##print(values)
# integer encode

#
## binary encode
#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# convert the labels from integers to vectors
Y_train = LabelBinarizer().fit_transform(Y_train)
Y_test = LabelBinarizer().fit_transform(Y_test)


num_features = X.shape[0]
training_length = X.shape[1]

#num_features = data.shape[0]
model = Sequential()

model.add(Embedding(num_features, 100, input_length=X.shape[1], trainable=False, mask_zero=True))
# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))
# Recurrent layer
model.add(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))

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

from sklearn.utils.multiclass import unique_labels
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
    classes = np.array(classes)[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(12,12))
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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

#print(classification_report(np.argmax(Y_test,axis=1), np.argmax(prediction,axis=1), target_names=class_names))

np.set_printoptions(precision=2)
prediction = (prediction > 0.5).astype('int')
# Plot non-normalized confusion matrix
plot_confusion_matrix(np.argmax(Y_test,axis=1),  np.argmax(prediction,axis=1), classes=classNames, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(np.argmax(Y_test,axis=1),  np.argmax(prediction,axis=1), classes=classNames, normalize=True, title='Normalized confusion matrix')

plt.show()



