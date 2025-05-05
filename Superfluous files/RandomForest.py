import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import time as clock

start = clock.time()

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

# n_trees = 500
# k=5
# kf = KFold(n_splits=k, shuffle=True, random_state=8)
# indeces = kf.split(g_tot)

accuracy_list = []
accuracy_std_list = []
sensitivity_list = []
sensitivity_std_list = []
specificity_list = []
specificity_std_list = []
n_trees_list = []

for trees in range(10,500, 20):
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []

    k=5
    kf = KFold(n_splits=k, shuffle=True, random_state=8)
    indeces = kf.split(g_tot)

    for i,(train_index, test_index) in enumerate(indeces):
        print('Fold:', i)    

        temp_accuracy = 0
        temp_sensitivity = 0
        temp_specificity = 0
        temp_NPos = 0
        temp_NNeg = 0

        train_pattern = g_tot[train_index]
        train_labels = label_tot[train_index]

        test_pattern = g_tot[test_index]
        test_labels = label_tot[test_index]

        for t_lab in test_labels:
            if t_lab == label1:
                temp_NPos += 1
            if t_lab == label0:
                temp_NNeg += 1

        model = RandomForestClassifier(n_estimators=trees, 
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

        # conf_matrix = confusion_matrix(test_labels, test_prediction, labels=[label0,label1])
        # print(conf_matrix)

        for p, pred in enumerate(test_prediction):  #label0 = 'NEGATIVI'    label1 = 'POSITIVI'
            if pred == label1 and test_labels[p] == label1:
                temp_accuracy += 1
                temp_sensitivity += 1

            if pred == label0 and test_labels[p] == label0:
                temp_accuracy += 1
                temp_specificity += 1

        temp_accuracy /= (temp_NNeg + temp_NPos)
        temp_sensitivity /= temp_NPos
        temp_specificity /= temp_NNeg

        fold_accuracy.append(temp_accuracy)
        fold_sensitivity.append(temp_sensitivity)
        fold_specificity.append(temp_specificity)

    accuracy = np.mean(fold_accuracy)
    accuracy_std = np.std(fold_accuracy)
    sensitivity = np.mean(fold_sensitivity)
    sensitivity_std = np.std(fold_sensitivity)
    specificity = np.mean(fold_specificity)
    specificity_std = np.std(fold_specificity)

    print(f'Risultati per {trees} alberi')
    print('Accuracy:', round(accuracy,3), '+/-', round(accuracy_std,3), '\nSensitivity:', round(sensitivity,3), '+/-', round(sensitivity_std,3), '\nSpecificity:', round(specificity,3), '+/-',round(specificity_std,3),'\n')

    n_trees_list.append(trees)
    accuracy_list.append(accuracy)
    accuracy_std_list.append(accuracy_std)
    sensitivity_list.append(sensitivity)
    sensitivity_std_list.append(sensitivity_std)
    specificity_list.append(specificity)
    specificity_std_list.append(specificity_std)


fig,ax = plt.subplots(3,1)
ax[0].scatter(n_trees_list,accuracy_list, color='blue')
ax[0].errorbar(n_trees_list,accuracy_list, yerr=accuracy_std_list, color='blue')
ax[0].set_title('Accuracy')
ax[0].grid(True)
ax[1].scatter(n_trees_list,sensitivity_list, color='green')
ax[1].errorbar(n_trees_list,sensitivity_list, yerr=sensitivity_std_list, color='green')
ax[1].set_title('Sensitivity')
ax[1].grid(True)
ax[2].scatter(n_trees_list,specificity_list, color='red')
ax[2].errorbar(n_trees_list,specificity_list, yerr=specificity_std_list, color='red')
ax[2].set_title('Specificity')
ax[2].grid(True)

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")
plt.show()