import numpy as np
import pandas as pd
import openpyxl
import sklearn.ensemble as ens
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.io import loadmat

path_file = "signal__b.mat"

data = loadmat(path_file)

g0 = data['g__0']
g1 = data['g__1']
tot_pattern = np.concatenate((g0, g1), axis=0)

Nneg = len(g0[:, 0])
Npos = len(g1[:, 0])

g0_labels = np.full(Nneg, "NEGATIVE")
g1_labels = np.full(Npos, "POSITIVE")
tot_labels = np.concatenate((g0_labels, g1_labels), axis=0)

fig, ax = plt.subplots()
ax.plot(g0[0])

plt.show()
plt.close(fig)

# Metrics
accuracy = 0.;
sensitivity = 0.;
specificity = 0.;

# Iperparameters
n_trees = 10
k = 5                   #k-fold

kf = KFold(n_splits = k, shuffle = True, random_state = 8)
indices = kf.split(tot_labels)


for train_index, test_index in indices:
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

    for j, pred in enumerate(test_prediction):
        if pred == "POSITIVE" and test_labels[j] == "POSITIVE":
            accuracy += 1./(Npos+Nneg)
            sensitivity += 1./Npos
    
        if pred == "NEGATIVE" and test_labels[j] == "NEGATIVE":
            accuracy += 1./(Npos+Nneg)
            specificity += 1./Nneg

print("Accuracy: {:.2%}".format(accuracy))
print("Sensitivity: {:.2%}".format(sensitivity))
print("Specificity: {:.2%}".format(specificity))
