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

k_fold = 0
'trial_truth_'
'trial_lie_'

ids = {
    'lawyer': ['trial_truth_001', 'trial_truth_002'],
    'andrea': ['trial_truth_003', 'trial_truth_004', 'trial_truth_005', 'trial_truth_006', 'trial_truth_007',
               'trial_lie_007', 'trial_lie_008', 'trial_lie_009', 'trial_lie_010', 'trial_lie_011', 'trial_lie_012',
               'trial_lie_013'],
    'jodi': ['trial_truth_008', 'trial_truth_009', 'trial_truth_010', 'trial_truth_011', 'trial_truth_012',
             'trial_truth_013', 'trial_truth_014', 'trial_truth_015', 'trial_truth_054', 'trial_truth_055',
             'trial_truth_056', 'trial_lie_014', 'trial_lie_015', 'trial_lie_016', 'trial_lie_017', 'trial_lie_018',
             'trial_lie_019', 'trial_lie_020', 'trial_lie_021', 'trial_lie_022', 'trial_lie_023', 'trial_lie_024',
             'trial_lie_025', 'trial_lie_026', 'trial_lie_027', 'trial_lie_028', 'trial_lie_029', 'trial_lie_030',
             'trial_lie_031', 'trial_lie_054', 'trial_lie_055', 'trial_lie_056', 'trial_lie_057'],
    'fernando': ['trial_truth_016', 'trial_truth_017'],
    'chris': ['trial_truth_018', 'trial_truth_019'],
    'steven': ['trial_truth_020'],
    'randy': ['trial_truth_021'],
    'marvin': ['trial_truth_022'],
    'alan': ['trial_truth_023'],
    'ken': ['trial_truth_024', 'trial_truth_025'],
    'mitchelle': ['trial_truth_026', 'trial_truth_049', 'trial_lie_035'],
    'martin': ['trial_truth_027'],
    'donna': ['trial_truth_028', 'trial_lie_032'],
    'bessman': ['trial_truth_029', 'trial_truth_031', 'trial_truth_032', 'trial_truth_033', 'trial_truth_034',
                'trial_truth_035', 'trial_truth_036', 'trial_truth_038'],
    'james': ['trial_truth_030', 'trial_truth_041', 'trial_truth_042', 'trial_truth_043'],
    'jonathan': ['trial_truth_037'],
    'jamie': ['trial_truth_039', 'trial_truth_040', 'trial_lie_033', 'trial_lie_034'],
    'charles': ['trial_truth_044', 'trial_truth_045', 'trial_truth_046', 'trial_truth_047', 'trial_truth_048'],
    'carlos': ['trial_truth_050', 'trial_lie_046', 'trial_lie_047', 'trial_lie_048', 'trial_lie_049'],
    'owen': ['trial_truth_051'],
    'edgar': ['trial_truth_052'],
    'crystal': ['trial_truth_053', 'trial_truth_060', 'trial_lie_037', 'trial_lie_038', 'trial_lie_039',
                'trial_lie_040'],
    'marissa': ['trial_lie_036', 'trial_lie_050', 'trial_lie_051'],
    'robert': ['trial_lie_041', 'trial_lie_042', 'trial_lie_043'],
    'dyches': ['trial_lie_044'],
    'kelly': ['trial_lie_045'],
    'scott': ['trial_lie_052'],
    'micheal': ['trial_lie_053'],
    'lance': ['trial_lie_058'],
    'candace': ['trial_lie_059', 'trial_lie_060', 'trial_lie_061'],
    'amanda': ['trial_truth_057', 'trial_truth_058', 'trial_truth_059', 'trial_lie_001', 'trial_lie_002',
               'trial_lie_003', 'trial_lie_004', 'trial_lie_005', 'trial_lie_006'],
}


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


def load(k_fold, random=False):
    l = 0
    for k, v in ids.items():
        l += len(v)

    print(l)

    speakers = list(ids.keys())
    print(len(speakers))

    with open('cv_10fold_index.pkl', 'rb') as f:
        speakers, cv = pickle.load(f)

    pprint(speakers)
    train_k = []
    test_k = []
    tr = cv[k_fold][0]
    te = cv[k_fold][1]
    print(tr, te)
    for x in tr:
        train_k.extend(ids[speakers[x]])
    for x in te:
        test_k += ids[speakers[x]]

    print(train_k)
    print(len(train_k))
    print(test_k)
    print(len(test_k))

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
    if random:
        from random import randint
        print("RANDOM")
        # print(randint(0, 9))
        for k in keys:
            if k in train_k:
                X_text_train.append(np.random.uniform(low=-100.0, high=100.0, size=(300,)))
                X_audio_train.append(np.random.uniform(low=-1.0, high=1.0, size=(300,)))
                X_video_train.append(np.random.uniform(low=-1.0, high=1.0, size=(300,)))
                X_gest_train.append(np.random.uniform(low=-1.0, high=1.0, size=(39,)))
                h = np.array([0, 0])
                h[randint(0, 1)] = 1
                Y_train.append(h)
                # print(k, labels[k])
            elif k in test_k:
                X_text_test.append(np.random.uniform(low=-1.0, high=1.0, size=(300,)))
                X_audio_test.append(np.random.uniform(low=-1.0, high=1.0, size=(300,)))
                X_video_test.append(np.random.uniform(low=-1.0, high=1.0, size=(300,)))
                X_gest_test.append(np.random.uniform(low=-1.0, high=1.0, size=(39,)))
                h = np.array([0, 0])
                h[randint(0, 1)] = 1
                # Y_train.append(h)
                Y_test.append(h)
            else:
                print(k, 'something wrong')

    else:
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
