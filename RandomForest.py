import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


path_file = '01.SVM - RF - Matlab [signal]/data/signal__b.mat'

data = loadmat(path_file)

# for key, values in data.items():
#     print(key,'\n',values,'\n\n')

g0 = data['g__0']
g1 = data['g__1']

NNeg = len(g0)
NPos = len(g1)

g_tot = np.concatenate((g0,g1), axis=0)

# fig, ax = plt.subplots()
# plt.plot(g0[10])
# plt.title('G0')

# fig1, ax1 = plt.subplots()
# plt.plot(g1[10])
# plt.title('G1')

# plt.show()
label0 = 'NEGATIVI'
label1 = 'POSITIVI'
label_g0 = np.full(NNeg, label0)
label_g1 = np.full(NPos, label1)
label_tot = np.concatenate((label_g0,label_g1), axis=0)

n_trees = 500
k=5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
indeces = kf.split(g_tot)

# #check if the k-folding shuffles mantains the correct labels
# for i,(train_index, test_index) in enumerate(indeces):
#     print('Fold:', i)
#     # print('Test:',test_index,'\nTrain:', train_index,'\nLabel:', label_tot[test_index])
#     print('Train')
#     for train in train_index:
#         print(train, label_tot[train])
#     print('Test')
#     for test in test_index:
#         print(test, label_tot[test])
#     print('\n')

accuracy = 0
sensitivity = 0
specificity = 0
false_positive = 0

for i,(train_index, test_index) in enumerate(indeces):
    print('Fold:', i)    

    train_pattern = g_tot[train_index]
    train_labels = label_tot[train_index]

    test_pattern = g_tot[test_index]
    test_labels = label_tot[test_index]

    model = RandomForestClassifier(n_estimators=n_trees, 
                                criterion='gini',
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='sqrt', 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                bootstrap=True)
    
    model.fit(train_pattern,train_labels)

    test_prediction = model.predict(test_pattern)
    print(test_prediction)

    conf_matrix = confusion_matrix(label_tot, test_prediction, labels=[label0, label1])
    print(conf_matrix)

    for p, pred in enumerate(test_prediction):  #label0 = 'NEGATIVI'    label1 = 'POSITIVI'
        if pred == label1 and test_labels[p] == label1:
            accuracy += 1
            sensitivity += 1

        if pred == label0 and test_labels[p] == label0:
            accuracy += 1
            specificity += 1

        if pred == label1 and test_labels[p] == label0:
            false_positive += 1

accuracy /= (NNeg + NPos)
sensitivity /= NPos
specificity /= (NNeg + false_positive)

print('\nRISULTATI')
print('Accuracy:', round(accuracy,3), '\nSensitivity:', round(sensitivity,3), '\nSpecificity:', round(specificity,3))