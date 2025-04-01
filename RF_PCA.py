import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import time as clock
import pandas as pd

from sklearn.model_selection import KFold
from scipy.io import loadmat
from scipy.linalg import eig

start = clock.time()

path_file = "signal__b.mat"

data = loadmat(path_file)

g0 = data['g__0']
g1 = data['g__1']

print("Calculating correlation matrix...")
tot_pattern = np.concatenate((g0, g1), axis=0)
corr = np.corrcoef(tot_pattern, rowvar=False)

print('Calculating eigenvalues and eigenvectors...')
e_val, e_vec = eig(corr) #in teoria pi� efficiente con scipy
e_vec = np.transpose(e_vec)

print('Sorting eigenvalues and eigenvectors...\n')
sort_index = np.argsort(e_val)[::-1]
e_val = e_val[sort_index].real()
e_vec = e_vec[sort_index,:].real()

for i in range(len(e_val[:20])):
    print('Eigenvalue: {}'.format(e_val[i]))
    print('Eigenvector: {}\n'.format(e_vec[i]))

e_val_sum = np.sum(e_val)

val_sum = 0
print('\nOriginal\tPercent\tCumulative')
for i,val in enumerate(e_val[:20]):
    val_sum += val
    print('{}\t{:.2%}\t{.2%}'.format(sort_index[i], val/e_val_sum, val_sum/e_val_sum))

df_eval = pd.DataFrame(data = e_val)
df_evec = pd.DataFrame(data = e_vec)
df_eval.to_csv('eigenvalues.csv', sep = ',')
df_evec.to_csv('eigenvectors.csv', sep = ',')

tot_pattern = np.concatenate((g0[:17], g1[:17]), axis=0)
ext_test_pattern = np.concatenate((g0[18:], g1[18:]), axis=0)

Nneg = len(g0[:, 0])
Npos = len(g1[:, 0])

Nneg_ext = int(len(ext_test_pattern[:, 0])/2)
Npos_ext = int(len(ext_test_pattern[:, 0])/2)

label0 = 'NEGATIVI'
label1 = 'POSITIVI'
g0_labels = np.full(Nneg, label0)
g1_labels = np.full(Npos, label1)
tot_labels = np.concatenate((g0_labels[:17], g1_labels[:17]), axis=0)
ext_test_labels = np.concatenate((g0_labels[18:], g1_labels[18:]), axis=0)

# print(len(tot_labels))
# print(tot_labels)
# print(len(ext_test_labels))
# print(ext_test_labels)

# Iperparameters
k = 5
n_trees = 100

# Metrics
fold_accuracy = []
fold_sensitivity = []
fold_specificity = []

kf = KFold(n_splits = k, shuffle = True, random_state = 8)
indices = kf.split(tot_labels)

vote_pos = np.zeros(len(ext_test_labels))
vote_neg = np.zeros(len(ext_test_labels))

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

    # External tests: Majority vote
    ext_prediction = model.predict(ext_test_pattern)

    for j, pred in enumerate(ext_prediction):
        if pred == label0:
            vote_neg[j] += 1
        if pred == label1:
            vote_pos[j] += 1

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

print(vote_neg)
print(vote_pos)

for i, true_label in enumerate(ext_test_labels):
    if vote_neg[i]>=vote_pos[i] and true_label == label0:
        ext_accuracy += 1./(Npos_ext+Nneg_ext)
        ext_specificity += 1./Nneg_ext
    if vote_neg[i]<vote_pos[i] and true_label == label1:
        ext_accuracy += 1./(Npos_ext+Nneg_ext)
        ext_sensitivity += 1./Npos_ext

print("Performance of external test")
print("Accuracy: {:.2%}".format(ext_accuracy))
print("Sensitivity: {:.2%}".format(ext_sensitivity))
print("Specificity: {:.2%}".format(ext_specificity))
print("\n")

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")