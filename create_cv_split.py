"""
Features:
1. Audio
2. Video
3. Textual
4. Gesture Features
"""
from random import shuffle
import csv
import pickle
from pprint import pprint

import numpy as np

k_fold = 1
'trial_truth_'
'trial_lie_'

ids = {
    'andrea': ['trial_truth_003', 'trial_truth_004', 'trial_truth_005', 'trial_truth_006', 'trial_truth_007',
               'trial_lie_007', 'trial_lie_008', 'trial_lie_009', 'trial_lie_010', 'trial_lie_011', 'trial_lie_012',
               'trial_lie_013'],
    'fernando': ['trial_truth_016', 'trial_truth_017'],
    'robert': ['trial_lie_041', 'trial_lie_042', 'trial_lie_043'],
    'chris': ['trial_truth_018', 'trial_truth_019'],
    'lance': ['trial_lie_058'],
    'dyches': ['trial_lie_044'],
    'steven': ['trial_truth_020'],
    'randy': ['trial_truth_021'],
    'candace': ['trial_lie_059', 'trial_lie_060', 'trial_lie_061'],
    'lawyer': ['trial_truth_001', 'trial_truth_002'],
    'marvin': ['trial_truth_022'],
    'jodi': ['trial_truth_008', 'trial_truth_009', 'trial_truth_010', 'trial_truth_011', 'trial_truth_012',
             'trial_truth_013', 'trial_truth_014', 'trial_truth_015', 'trial_truth_054', 'trial_truth_055',
             'trial_truth_056', 'trial_lie_014', 'trial_lie_015', 'trial_lie_016', 'trial_lie_017', 'trial_lie_018',
             'trial_lie_019', 'trial_lie_020', 'trial_lie_021', 'trial_lie_022', 'trial_lie_023', 'trial_lie_024',
             'trial_lie_025', 'trial_lie_026', 'trial_lie_027', 'trial_lie_028', 'trial_lie_029', 'trial_lie_030',
             'trial_lie_031', 'trial_lie_054', 'trial_lie_055', 'trial_lie_056', 'trial_lie_057'],
    'alan': ['trial_truth_023'],
    'kelly': ['trial_lie_045'],
    'ken': ['trial_truth_024', 'trial_truth_025'],
    'scott': ['trial_lie_052'],
    'micheal': ['trial_lie_053'],
    'mitchelle': ['trial_truth_026', 'trial_truth_049', 'trial_lie_035'],
    'martin': ['trial_truth_027'],
    'donna': ['trial_truth_028', 'trial_lie_032'],
    'bessman': ['trial_truth_029', 'trial_truth_031', 'trial_truth_032', 'trial_truth_033', 'trial_truth_034',
                'trial_truth_035', 'trial_truth_036', 'trial_truth_038'],
    'marissa': ['trial_lie_036', 'trial_lie_050', 'trial_lie_051'],
    'james': ['trial_truth_030', 'trial_truth_041', 'trial_truth_042', 'trial_truth_043'],

    'jonathan': ['trial_truth_037'],
    'jamie': ['trial_truth_039', 'trial_truth_040', 'trial_lie_033', 'trial_lie_034'],

    'carlos': ['trial_truth_050', 'trial_lie_046', 'trial_lie_047', 'trial_lie_048', 'trial_lie_049'],
    'owen': ['trial_truth_051'],
    'edgar': ['trial_truth_052'],
    'crystal': ['trial_truth_053', 'trial_truth_060', 'trial_lie_037', 'trial_lie_038', 'trial_lie_039',
                'trial_lie_040'],
    'charles': ['trial_truth_044', 'trial_truth_045', 'trial_truth_046', 'trial_truth_047', 'trial_truth_048'],
    'amanda': ['trial_truth_057', 'trial_truth_058', 'trial_truth_059', 'trial_lie_001', 'trial_lie_002',
               'trial_lie_003', 'trial_lie_004', 'trial_lie_005', 'trial_lie_006'],
}
l = 0
for k, v in ids.items():
    l += len(v)

# print(l)

speakers = list(ids.keys())
# shuffle(speakers)
print(len(speakers))
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
a = kf.get_n_splits(speakers)
print(a)
# print(kf.split(list(speakers)))

cv = {}
for i, (train_index, test_index) in enumerate(kf.split(list(speakers))):
    print("\nCV%s:" % (i + 1), train_index, test_index)
    cv[i] = [list(train_index), list(test_index)]
    # if i == 1:
    tr_lie_cnt = 0
    tr_true_cnt = 0
    te_lie_cnt = 0
    te_true_cnt = 0
    for idx in list(train_index):
        for j in ids[speakers[idx]]:
            if 'lie' in j:
                tr_lie_cnt += 1
            elif 'truth' in j:
                tr_true_cnt += 1
            else:
                print('train_index: something wrong')

    for idx in list(test_index):
        for j in ids[speakers[idx]]:
            if 'lie' in j:
                te_lie_cnt += 1
            elif 'truth' in j:
                te_true_cnt += 1
            else:
                print('test_index: something wrong')
    print("tr_lie_cnt", tr_lie_cnt)
    print("tr_true_cnt", tr_true_cnt)
    print("te_lie_cnt", te_lie_cnt)
    print("te_true_cnt", te_true_cnt)
# print(cv)

with open('cv_10fold_index.pkl', 'wb') as f:
    pickle.dump([speakers, cv], f)

for k_fold in range(10):
    train_k = []
    test_k = []
    tr = cv[k_fold][0]
    te = cv[k_fold][1]
#     print(tr, te)
    for x in tr:
        train_k.extend(ids[speakers[x]])
    for x in te:
        test_k += ids[speakers[x]]
    print("CV %d, %d:%d" % (k_fold, len(train_k), len(test_k)))

