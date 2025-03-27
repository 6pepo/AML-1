import numpy as np
# import pandas as pd
# import openpyxl
import sklearn.ensemble as ens
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time as clock

start = clock.time()

path_file = "signal__b.mat"

data = loadmat(path_file)

g0 = data['g__0']
g1 = data['g__1']
tot_pattern = np.concatenate((g0[:17], g1[:17]), axis=0)
ext_test_pattern = np.concatenate((g0[18:], g1[18:]), axis=0)

Nneg = len(g0[:, 0])
Npos = len(g1[:, 0])

label0 = 'NEGATIVI'
label1 = 'POSITIVI'
g0_labels = np.full(Nneg, label0)
g1_labels = np.full(Npos, label1)
tot_labels = np.concatenate((g0_labels[:17], g1_labels[:17]), axis=0)
ext_test_labels = np.concatenate((g0_labels[18:], g1_labels[18:]), axis=0)

print(len(tot_labels))
print(len(ext_test_labels))

# Iperparameters
k = 5
n_trees = 150

# Metrics
fold_accuracy = [];
fold_sensitivity = [];
fold_specificity = [];

kf = KFold(n_splits = k, shuffle = True, random_state = 8)
indices = kf.split(tot_labels)


for i, (train_index, test_index) in enumerate(indices):

    #print("Fold: {}".format(i+1))

    train_pattern = tot_pattern[train_index]
    train_labels = tot_labels[train_index]

    test_pattern = tot_pattern[test_index]
    test_labels = tot_labels[test_index]



    model = ens.RandomForestClassifier(n_estimators=n_trees, 
                                               criterion='gini',
                                               max_depth=None,
                                               min_samples_split=2,
                                               min_samples_leaf=1, 
                                               min_weight_fraction_leaf=0.0, 
                                               max_features='sqrt', 
                                               max_leaf_nodes=None, 
                                               min_impurity_decrease=0.0, 
                                               bootstrap=True)

    model.fit(train_pattern, train_labels)

    test_prediction = model.predict(test_pattern)

    temp_Npos = 0
    temp_Nneg = 0

    for t_lab in test_labels:
        if t_lab == label1:
            temp_Npos += 1
        if t_lab == label0:
            temp_Nneg += 1

    temp_accuracy = 0.
    temp_sensitivity = 0.
    temp_specificity = 0.

    for j, pred in enumerate(test_prediction):
        if pred == label1 and test_labels[j] == label1:
            temp_accuracy += 1./(temp_Npos+temp_Nneg)
            temp_sensitivity += 1./temp_Npos
    
        if pred == label0 and test_labels[j] == label0:
            temp_accuracy += 1./(temp_Npos+temp_Nneg)
            temp_specificity += 1./temp_Nneg

    fold_accuracy.append(temp_accuracy)
    fold_sensitivity.append(temp_sensitivity)
    fold_specificity.append(temp_specificity)


print("Fold\tAcc\tSensit\tSpecif")
for i, acc in enumerate(fold_accuracy):
    print("{}\t{:.2%}\t{:.2%}\t{:.2%}".format(i+1, acc, fold_sensitivity[i], fold_specificity[i]))

print("\n")

accuracy = np.mean(fold_accuracy)
accuracy_std = np.std(fold_accuracy)
sensitivity = np.mean(fold_sensitivity)
sensitivity_std = np.std(fold_sensitivity)
specificity = np.mean(fold_specificity)
specificity_std = np.std(fold_specificity)

acc_rel = accuracy_std/accuracy
sens_rel = sensitivity_std/sensitivity
spec_rel = specificity_std/specificity

print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(accuracy, accuracy_std, acc_rel))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(sensitivity, sensitivity_std, sens_rel))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(specificity, specificity_std, spec_rel))
