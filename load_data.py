"""
Features:
1. Audio
2. Video
3. Textual
4. Gesture Features
"""
import csv
import pickle
from pprint import pprint

import numpy as np

np.random.seed(1234)
train_lie_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 46, 47, 48, 49, 59, 60, 61, 54, 55, 56, 57]
# train_lie_k = train_lie_k[:-2]
test_lie_k = list(set(range(1, 62)) - set(train_lie_k))

train_truth_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 54, 55, 56, 26, 49, 28, 39, 40, 50, 53, 60, 57, 58,
                 59, 16, 17, 18, 19, 29, 31, 32, 33, 34, 35, 36, 38, 30, 41, 42, 43, 44, 45, 46, 47, 48]
# train_truth_k = train_truth_k[:-2]
test_truth_k = list(set(range(1, 61)) - set(train_truth_k))

print(test_lie_k, test_truth_k)
print(len(train_lie_k), len(train_truth_k))
print(len(test_lie_k), len(test_truth_k))

train_k = []
test_k = []

for k1, k2 in zip(train_truth_k, train_lie_k):
    t = 'trial_truth_%03d' % k1
    l = 'trial_lie_%03d' % k2
    # print(t, l)
    train_k.append(t)
    train_k.append(l)

for k1, k2 in zip(test_truth_k, test_lie_k):
    t = 'trial_truth_%03d' % k1
    l = 'trial_lie_%03d' % k2
    # print(t, l)
    test_k.append(t)
    test_k.append(l)
test_k.append('trial_lie_%03d' % test_lie_k[-1])
# print(test_k)
print(len(set(test_k)))


def load_text_features():
    features = pickle.load(open('create_data/text/deception_text_features.pickle', 'rb'))
    print(features['trial_truth_056']['features'].shape)
    return features


def load_audio_features():
    features = pickle.load(open('create_data/audio/deception_features.pickle', 'rb'))
    print(features['trial_truth_056']['features'].shape)
    return features


def load_video_features():
    (intermediate_output_train, y_train, intermediate_output_test, y_test, train_key, test_key) = pickle.load(
        open('create_data/video/features_new.pickle', 'rb'))
    # print(intermediate_output_train.shape)
    # print(intermediate_output_test.shape)
    # print(len(train_key))
    # print(len(test_key))
    feats = np.append(intermediate_output_train, intermediate_output_test, axis=0)
    keys = train_key + test_key
    # print(feats.shape)
    # print(keys)
    features = {}
    for i, k in enumerate(keys):
        features[k] = feats[i]
    # print(features['trial_truth_056'].shape)
    return features


def load_gesture_features():
    labels = ['OtherGestures', 'Smile', 'Laugh', 'Scowl', 'otherEyebrowMovement', 'Frown', 'Raise',
              'OtherEyeMovements', 'Close-R', 'X-Open', 'Close-BE', 'gazeInterlocutor', 'gazeDown', 'gazeUp',
              'otherGaze', 'gazeSide', 'openMouth', 'closeMouth', 'lipsDown', 'lipsUp', 'lipsRetracted',
              'lipsProtruded', 'SideTurn', 'downR', 'sideTilt', 'backHead', 'otherHeadM', 'sideTurnR', 'sideTiltR',
              'waggle', 'forwardHead', 'downRHead', 'singleHand', 'bothHands', 'otherHandM', 'complexHandM',
              'sidewaysHand', 'downHands', 'upHands']
    # print(len(labels))
    data = {}
    with open("create_data/All_Gestures_Deceptive and Truthful.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            f = []
            for r in labels:
                f.append(int(row[r]))
            data[row['id'][:-4]] = np.array(f)
    # print(data)
    # print(data['trial_truth_060'].shape)
    # print(len(data))
    return data


def load_labels():
    labels = {}
    with open("create_data/All_Gestures_Deceptive and Truthful.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (row['class'] == 'deceptive'):
                labels[row['id'].rsplit('.', 1)[0]] = np.array([0, 1])
            else:
                labels[row['id'].rsplit('.', 1)[0]] = np.array([1, 0])
    # pprint(labels)
    return labels


def load():
    text_feat = load_text_features()
    gesture_feat = load_gesture_features()
    video_feat = load_video_features()
    audio_feat = load_audio_features()
    labels = load_labels()
    X_text_train = []
    X_gest_train = []
    X_audio_train = []
    X_video_train = []
    Y_train = []
    X_text_test = []
    X_gest_test = []
    X_audio_test = []
    X_video_test = []
    Y_test = []
    keys = list(labels.keys())
    np.random.shuffle(keys)
    # print(keys)
    """
    for k in train_k:
        X_text_train.append(text_feat[k]['features'])
        X_audio_train.append(audio_feat[k]['features'])
        X_video_train.append(video_feat[k])
        X_gest_train.append(gesture_feat[k])
        Y_train.append(labels[k])

    for k in test_k:
        X_text_test.append(text_feat[k]['features'])
        X_audio_test.append(audio_feat[k]['features'])
        X_video_test.append(video_feat[k])
        X_gest_test.append(gesture_feat[k])
        Y_test.append(labels[k])
    """
    for k in keys:
        if k in train_k:
            X_text_train.append(text_feat[k]['features'])
            X_audio_train.append(audio_feat[k]['features'])
            X_video_train.append(video_feat[k])
            X_gest_train.append(gesture_feat[k])
            Y_train.append(labels[k])
            # print(k, labels[k])
        elif k in test_k:
            X_text_test.append(text_feat[k]['features'])
            X_audio_test.append(audio_feat[k]['features'])
            X_video_test.append(video_feat[k])
            X_gest_test.append(gesture_feat[k])
            Y_test.append(labels[k])
        else:
            print(k, 'something wrong')

    X_text_train = np.array(X_text_train)
    X_audio_train = np.array(X_audio_train)
    X_gest_train = np.array(X_gest_train)
    X_video_train = np.array(X_video_train)
    Y_train = np.array(Y_train)
    X_text_test = np.array(X_text_test)
    X_audio_test = np.array(X_audio_test)
    X_gest_test = np.array(X_gest_test)
    X_video_test = np.array(X_video_test)
    Y_test = np.array(Y_test)
    print(X_text_train.shape, X_text_test.shape)
    return X_text_train, X_text_test, X_audio_train, X_audio_test, X_gest_train, X_gest_test, X_video_train, X_video_test, Y_train, Y_test

# print(load())
# load()
