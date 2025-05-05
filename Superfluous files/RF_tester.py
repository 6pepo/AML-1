import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import time as clock

from sklearn.model_selection import KFold
from scipy.io import loadmat

start = clock.time()

# Training and internal testing set
path_file = "signal__b.mat"

data = loadmat(path_file)

g0 = data['g__0']
g1 = data['g__1']
tot_pattern = np.concatenate((g0, g1), axis=0)

Nneg = len(g0[:, 0])
Npos = len(g1[:, 0])

label0 = 'NEGATIVI'
label1 = 'POSITIVI'
g0_labels = np.full(Nneg, label0)
g1_labels = np.full(Npos, label1)
tot_labels = np.concatenate((g0_labels, g1_labels), axis=0)

# External testing set
path_file = 'signal__a.mat'

data = loadmat(path_file)

g0_ext = data['g__0']
g1_ext = data['g__1']
tot_ext_patt = np.concatenate((g0_ext, g1_ext), axis=0)

Nneg_ext = len(g0_ext)
Npos_ext = len(g1_ext)

g0_ext_labels = np.full(Nneg_ext, label0)
g1_ext_labels = np.full(Npos_ext, label1)
ext_labels = np.concatenate((g0_ext_labels, g1_ext_labels), axis=0)


# Iperparameters
k = 5
n_trees = 100

models = []

# Metrics
fold_accuracy = []
fold_sensitivity = []
fold_specificity = []

kf = KFold(n_splits = k, shuffle = True, random_state = 8)
indices = kf.split(tot_labels)

vote_pos_ext = np.zeros(len(ext_labels))
vote_neg_ext = np.zeros(len(ext_labels))

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
    models.append(model)

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

    # External test: Majority vote
    ext_prediction = model.predict(tot_ext_patt)

    for j, pred in enumerate(ext_prediction):
        if pred == label0:
            vote_neg_ext[j] += 1
        if pred == label1:
            vote_pos_ext[j] += 1

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

print("Performance of cross validation")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(accuracy, accuracy_std, acc_rel))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(sensitivity, sensitivity_std, sens_rel))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(specificity, specificity_std, spec_rel))
print("\n")

# Performance of External Test

ext_accuracy = 0.
ext_sensitivity = 0.
ext_specificity = 0.

# print(ext_labels) # Che siano in realtà invertiti?
print(vote_neg_ext)
print(vote_pos_ext)

for i, true_label in enumerate(ext_labels):
    if vote_neg_ext[i]>=vote_pos_ext[i] and true_label == label0:
        ext_accuracy += 1./(Npos_ext+Nneg_ext)
        ext_specificity += 1./Nneg_ext
    if vote_neg_ext[i]<vote_pos_ext[i] and true_label == label1:
        ext_accuracy += 1./(Npos_ext+Nneg_ext)
        ext_sensitivity += 1./Npos_ext

print("Performance of external test")
print("Accuracy: {:.2%}".format(ext_accuracy))
print("Sensitivity: {:.2%}".format(ext_sensitivity))
print("Specificity: {:.2%}".format(ext_specificity))
print("\n")


# Counting root feature in decision Trees
feat_counter = np.zeros(len(g0[0]))

for model in models:
    for estim in model.estimators_:
        feat_counter[estim.tree_.feature[0]] += 1

prim_feat_trees = np.argsort(feat_counter)[::-1]
print("Most frequent argument in tree roots:")
print("F_num\tCount")
for i in range(10):
    print("{}\t{}".format(prim_feat_trees[i], feat_counter[prim_feat_trees[i]]))
print(np.sum(feat_counter))

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")