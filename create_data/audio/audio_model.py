import csv
import io
from pprint import pprint

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.utils import np_utils
from tqdm import tqdm

import keras

# use 6k features from MOSI and train the MLP
# use the layer to extract the features
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model

np.random.seed(1000)


def load_data():
    deceptive_data = pd.read_pickle('features_Deceptive.pkl')
    truthful_data = pd.read_pickle('features_Truthful.pkl')
    deceptive_data['label'] = np.array([1] * deceptive_data.shape[0])
    print(deceptive_data.shape)
    truthful_data['label'] = np.array([0] * truthful_data.shape[0])
    print(truthful_data.shape)
    data = pd.DataFrame(columns=['file', 'features'])
    label = []
    for i in range(60):
        # print(deceptive_data.iloc[i]['file'])
        # print(deceptive_data.iloc[i]['features'])
        d = pd.DataFrame({'file': deceptive_data.iloc[i]['file'], 'features': [deceptive_data.iloc[i]['features']]})
        data = data.append(d)
        label.append(1)
        d = pd.DataFrame({'file': truthful_data.iloc[i]['file'], 'features': [truthful_data.iloc[i]['features']]})
        data = data.append(d)
        label.append(0)
    d = pd.DataFrame({'file': deceptive_data.iloc[60]['file'], 'features': [deceptive_data.iloc[60]['features']]})
    data = data.append(d)
    label.append(1)
    # data = truthful_data.append(deceptive_data)
    print(data.shape)
    print("len of Labels", len(label))
    print("Labels", label)
    # print(data)

    data = data.set_index('file')
    d = {}
    train_data = []
    for i, x in data.iterrows():
        # print i, i + '_' + x['split_id']
        l = list(x['features'])
        l = list(map(float, l))
        # print len(l)
        d[i] = l
        train_data.append(l)
    train_data = np.array(train_data)
    X_train = train_data[:90]
    X_test = train_data[90:]
    Y_train = label[:90]
    Y_test = label[90:]
    print(X_train.shape, X_test.shape)
    print(len(Y_train), len(Y_test))

    features_data = truthful_data.append(deceptive_data)

    features_data = features_data.set_index('file')
    d = {}
    feat_data = []
    for i, x in features_data.iterrows():
        # print i, i + '_' + x['split_id']
        l = list(x['features'])
        l = list(map(float, l))
        # print len(l)
        d[i] = l
        feat_data.append(l)
    feat_data = np.array(feat_data)
    print("X_data", feat_data.shape)
    # print(list(d.keys()))
    # print(d['trial_lie_005'])
    # print(d['trial_lie_005'])
    # print(feat_data)
    # print(X_data.shape)
    print(type(feat_data))

    return feat_data, X_train, X_test, Y_train, Y_test
    # return None, None, None


feat_data, X_train, X_test, Y_train, Y_test = load_data()
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
# print(Y_train, Y_test)

scale = np.amax(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = Y_train.shape[1]

model = Sequential()
model.add(Dense(5000, input_dim=input_dim))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(300, name='feature_output'))
model.add(Activation('tanh'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
# model.add(Activation('softmax'))
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=20, verbose=1)
]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
# model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2)

model.fit(X_train, Y_train, verbose=1, shuffle=True, epochs=1000, batch_size=30, callbacks=callbacks,
          validation_split=0.2)

score = model.evaluate(X_test, Y_test, batch_size=25)
print(score)
print("\n\nAcc: ", score[1])
# model.save('audio_mlp.model')

# model = load_model('audio_mlp.model')
"""
feature_model = Model(model.input, model.get_layer('feature_output').output)
features = feature_model.predict(feat_data)
# print features
print(features.shape)
print(features[1].shape, features[1])
deceptive_data = pd.read_pickle('features_Deceptive.pkl')
truthful_data = pd.read_pickle('features_Truthful.pkl')
deceptive_data['label'] = np.array([1] * deceptive_data.shape[0])
print(deceptive_data.shape)
truthful_data['label'] = np.array([0] * truthful_data.shape[0])
print(truthful_data.shape)
data = truthful_data.append(deceptive_data)
print(data.shape)
data = data.set_index('file', drop=False)
final_data = {}
y = 0
for i, x in data.iterrows():
    d = {}
    # print(i)
    # data = data.set_value(i, 'features', [features[i]])
    d['features'] = features[y]
    d['label'] = x['label']
    final_data[i] = d
    y += 1

# pprint(final_data)
import pickle

# with open('deception_features.pickle', 'wb') as handle:
#     pickle.dump(final_data, handle, protocol=2)
"""
