import pickle

import cv2
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential

# np.random.seed(12345)
np.random.seed(1000000)  # best 72%
# np.random.seed(0000000)  # best 66%

max_frames = 30
# image specification
img_rows, img_cols, img_depth = 96, 96, max_frames

# video_path = '../MOSI/Video/Segmented/'
# Training data
X_tr = {}  # variable to store entire dataset\

# with open('traintest.pickle', 'rb') as handle:
#     (train, test) = pickle.load(handle)
# print(len(train))
# print(len(test))

with open('deceptive_cropped_videos.p', 'rb') as handle:
    (deceptive_videos, deceptive_names) = pickle.load(handle)

with open('truthful_cropped_videos.p', 'rb') as handle:
    (truthful_videos, truthful_names) = pickle.load(handle)

max_f = 0

for i in range(len(deceptive_videos)):
    max_f = max(max_f, len(deceptive_videos[i]))

for i in range(len(truthful_videos)):
    max_f = max(max_f, len(truthful_videos[i]))

Vid = []
pad = np.zeros((96, 96))

for i in range(len(deceptive_videos)):
    # print i
    vid = []
    ctr = 0
    for j in range(min(len(deceptive_videos[i]), max_frames)):
        ctr += 1
        frame = cv2.cvtColor(deceptive_videos[i][j], cv2.COLOR_BGR2GRAY)
        vid.append(frame)
    for j in range(ctr, max_frames):
        vid.append(pad)
    Vid.append(vid)

vidd = np.asarray(Vid)

for i in range(vidd.shape[0]):
    X_tr[deceptive_names[i].rsplit('.', 1)[0]] = np.rollaxis(np.rollaxis(vidd[i], 2, 0), 2, 0)

Vid = []

for i in range(len(truthful_videos)):
    # print i
    vid = []
    ctr = 0
    for j in range(min(len(truthful_videos[i]), max_frames)):
        ctr += 1
        frame = cv2.cvtColor(truthful_videos[i][j], cv2.COLOR_BGR2GRAY)
        vid.append(frame)
    for j in range(ctr, max_frames):
        vid.append(pad)
    Vid.append(vid)

vidd = np.asarray(Vid)

for i in range(vidd.shape[0]):
    X_tr[truthful_names[i].rsplit('.', 1)[0]] = np.rollaxis(np.rollaxis(vidd[i], 2, 0), 2, 0)
    # print names[i].rsplit('.', 1)[0], X_tr[names[i].rsplit('.', 1)[0]].shape
# train = []
# test = []
na = []
for x in range(60):
    na.append(deceptive_names[x].rsplit('.', 1)[0])
    na.append(truthful_names[x].rsplit('.', 1)[0])
na.append(deceptive_names[60].rsplit('.', 1)[0])
# print(na)
train = na[:100]
test = na[100:]
# print("TRAIN: ", train, len(train))
# print("TEST: ", test, len(test))
'''
#Reading MOSI class
print video_path
listing = os.listdir(video_path)
print listing
max_frames = 0

ctr=0
for vid in listing:
    if(".mp4" in vid):
        ctr+=1
        if(ctr%10==0):
            print ctr
        name = vid.split(".")
        vid = video_path+vid
        #print name
        frames = []
        cap = cv2.VideoCapture(vid)
        fps = cap.get(5)
        total_frames = cap.get(7)
        if(total_frames>max_frames):
            max_frames = total_frames
        i = 0
        #print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
        for k in xrange(int(total_frames)):
            ret, frame = cap.read()

            if(ret):
                frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                print frames
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
#        cv2.destroyAllWindows()
        input1=np.array(frames)
        print input1.shape
        ipt=np.rollaxis(np.rollaxis(input1,2,0),2,0)

        X_tr[name[0]] = ipt


max_frames = int(max_frames)
'''
import csv

labels = {}
with open(
        "/projects/Datasets/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:

        if (row['class'] == 'deceptive'):
            labels[row['id'].rsplit('.', 1)[0]] = int(1)
        else:
            labels[row['id'].rsplit('.', 1)[0]] = int(0)
# print(X_tr)
# print("labels", labels.values())
'''
print "reached here"
max_frames = 1575

with open('xtr.pickle', 'rb') as handle:
        (X_tr) = pickle.load(handle)
print "done uploading"
'''
print("MODEL START")
from keras import backend as K

K.set_image_dim_ordering('th')
X_tr_array = {}
num_samples = 0
result = np.zeros((img_rows, img_cols, max_frames))
# ctr = 0
for key, value in X_tr.items():
    # ctr += 1
    # print ctr
    value = np.resize(value, result.shape)
    X_tr_array[key] = np.array(value)
    num_samples += len(X_tr_array[key])
    # if ctr==50:
    #    	break
    # convert the frames read into array
    # print(value.shape)
# print(num_samples)

train_data = [[], []]
test_data = [[], []]

train_key = []
test_key = []
# print(train, test)
for key in X_tr_array.keys():
    # print(key)
    if key in train:
        train_data[0].append(X_tr_array[key])
        train_data[1].append(labels[key])
        train_key.append(key)
    elif key in test:
        test_data[0].append(X_tr_array[key])
        test_data[1].append(labels[key])
        test_key.append(key)
    else:
        print("gone wrong")

train_data = np.array(train_data)
test_data = np.array(test_data)
(X_train, y_train) = (train_data[0], train_data[1])
(X_test, y_test) = (test_data[0], test_data[1])

Y_train = np.zeros((y_train.shape[0], 2))
Y_test = np.zeros((y_test.shape[0], 2))

for i in range(y_train.shape[0]):
    Y_train[i, y_train[i]] = 1
for i in range(y_test.shape[0]):
    Y_test[i, y_test[i]] = 1
# for x, y in zip(train_key, Y_train):
#     print(x, y)
# for x, y in zip(test_key, Y_test):
#     print(x, y)

train_set = np.zeros((len(train_key), 1, img_rows, img_cols, max_frames))
test_set = np.zeros((len(test_key), 1, img_rows, img_cols, max_frames))

for h in range(len(train_key)):
    train_set[h][0][:][:][:] = X_train[h]
for h in range(len(test_key)):
    test_set[h][0][:][:][:] = X_test[h]

patch_size = 30  # img_depth or number of frames used for each video
print('X_Train shape:', X_train.shape)
print(train_set.shape, 'train samples')
print('X_Test shape:', X_test.shape)
print(test_set.shape, 'test samples')

# convert class vectors to binary class matrices
print(Y_train.shape, Y_test.shape)
# Y_train = np_utils.to_categorical(y_train, nb_classes)


# Pre-processing

train_set = train_set.astype('float32')
train_set -= np.mean(train_set)
train_set /= np.max(train_set)

test_set = test_set.astype('float32')
test_set -= np.mean(test_set)
test_set /= np.max(test_set)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


# CNN Training parameters

batch_size = 15
nb_classes = 2
# Define model
# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = (3, 3, 3)

from keras.layers import Activation

model = Sequential()
model.add(Conv3D(nb_filters[0], nb_conv, input_shape=(1, img_rows, img_cols, max_frames), activation='relu'))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(300, kernel_initializer='normal', activation='relu', name="utter"))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, kernel_initializer='normal'))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# from keras.models import load_model

# del model  # deletes the existing model

nb_epoch = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(train_set, Y_train, verbose=1, validation_split=0.1, batch_size=batch_size, epochs=nb_epoch, shuffle=True,
          callbacks=[early_stopping])
# model.evaluate(X_test, Y_test, batch_size=128)

model.save('Deception_video_model_new_1.h5')  # creates a HDF5 file 'my_model.h5'

from keras.models import Model

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("utter").output)
intermediate_output_train = intermediate_layer_model.predict(train_set)
intermediate_output_test = intermediate_layer_model.predict(test_set)

score = model.evaluate(test_set, Y_test, batch_size=batch_size, verbose=1, sample_weight=None)
print()
print(score)
print("\nScore:", score[1] * 100)
train_preds = model.predict(train_set, verbose=0)
test_preds = model.predict(test_set, verbose=0)
print(intermediate_output_train.shape)
print(intermediate_output_test.shape)

import pickle

with open('./features_new_1.pickle', 'wb') as handle:
    pickle.dump((intermediate_output_train, y_train, intermediate_output_test, y_test, train_key, test_key), handle,
                protocol=pickle.HIGHEST_PROTOCOL)
