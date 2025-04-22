import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import time as clock
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold
from scipy.io import loadmat

start = clock.time()

n_iter = 100

# Iperparameters non-PCA
k = 7
n_trees = 245

# Iperparameters PCA
k_PCA = 5
n_trees_PCA = 150


    #NON PCA

# Training set
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
path_file = "signal__a.mat"

data = loadmat(path_file)

g0_ext = data['g__0']
g1_ext = data['g__1']
tot_ext_patt = np.concatenate((g0_ext, g1_ext), axis=0)

Nneg_ext = len(g0_ext)
Npos_ext = len(g1_ext)

g0_ext_labels = np.full(Nneg_ext, label1)                           #Labels are inverted!
g1_ext_labels = np.full(Npos_ext, label0)
ext_labels = np.concatenate((g0_ext_labels, g1_ext_labels), axis=0)

feat_counter = np.zeros(len(g0[0]))

acc_list = []
sens_list = []
spec_list = []

acc_ext_list = []
sens_ext_list = []
spec_ext_list = []

base_acc_bacc_list = []
base_sens_bacc_list = []
base_spec_bacc_list = []

base_acc_bsens_list = []
base_sens_bsens_list = []
base_spec_bsens_list = []

base_acc_bspec_list = []
base_sens_bspec_list = []
base_spec_bspec_list = []

print('Starting Non-PCA iterations...')
#start_non_pca = clock.process_time()                               #CPU time is different in 2.7-
for n in range(n_iter):
    # print("Iteration: {}/{} \r".format(n+1, n_iter)),
    print(f'Iteration: {n+1}/{n_iter}', end='\r' )

    models = []

    # Metrics
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []

    kf = KFold(n_splits = k, shuffle = True)
    indices = kf.split(tot_labels)

    vote_pos_ext = np.zeros(len(ext_labels))
    vote_neg_ext = np.zeros(len(ext_labels))

    for i, (train_index, test_index) in enumerate(indices):

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

    acc_list.append(np.mean(fold_accuracy))
    sens_list.append(np.mean(fold_sensitivity))
    spec_list.append(np.mean(fold_specificity))


    # Performance of External Test: Best in Training

    bacc_model = models[np.argmax(fold_accuracy)]
    bacc_accuracy = 0.
    bacc_sensitivity = 0.
    bacc_specificity = 0.
    bacc_predict = bacc_model.predict(tot_ext_patt)

    for j, pred in enumerate(bacc_predict):
        if pred == label1 and ext_labels[j] == label1:
            bacc_accuracy += 1./(Npos_ext+Nneg_ext)
            bacc_specificity += 1./Nneg_ext
    
        if pred == label0 and ext_labels[j] == label0:
            bacc_accuracy += 1./(Npos_ext+Nneg_ext)
            bacc_sensitivity += 1./Npos_ext

    base_acc_bacc_list.append(bacc_accuracy)
    base_sens_bacc_list.append(bacc_sensitivity)
    base_spec_bacc_list.append(bacc_specificity)


    bsens_model = models[np.argmax(fold_sensitivity)]
    bsens_accuracy = 0.
    bsens_sensitivity = 0.
    bsens_specificity = 0.
    bsens_predict = bsens_model.predict(tot_ext_patt)

    for j, pred in enumerate(bsens_predict):
        if pred == label1 and ext_labels[j] == label1:
            bsens_accuracy += 1./(Npos_ext+Nneg_ext)
            bsens_specificity += 1./Nneg_ext
    
        if pred == label0 and ext_labels[j] == label0:
            bsens_accuracy += 1./(Npos_ext+Nneg_ext)
            bsens_sensitivity += 1./Npos_ext

    base_acc_bsens_list.append(bsens_accuracy)
    base_sens_bsens_list.append(bsens_sensitivity)
    base_spec_bsens_list.append(bsens_specificity)


    bspec_model = models[np.argmax(fold_specificity)]
    bspec_accuracy = 0.
    bspec_sensitivity = 0.
    bspec_specificity = 0.
    bspec_predict = bspec_model.predict(tot_ext_patt)

    for j, pred in enumerate(bspec_predict):
        if pred == label1 and ext_labels[j] == label1:
            bspec_accuracy += 1./(Npos_ext+Nneg_ext)
            bspec_specificity += 1./Nneg_ext
    
        if pred == label0 and ext_labels[j] == label0:
            bspec_accuracy += 1./(Npos_ext+Nneg_ext)
            bspec_sensitivity += 1./Npos_ext

    base_acc_bspec_list.append(bspec_accuracy)
    base_sens_bspec_list.append(bspec_sensitivity)
    base_spec_bspec_list.append(bspec_specificity)


    # Performance of External Test: Majority Vote

    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.


    for i, true_label in enumerate(ext_labels):
        if vote_neg_ext[i]>=vote_pos_ext[i] and true_label == label0:
            ext_accuracy += 1./(Npos_ext+Nneg_ext)
            ext_specificity += 1./Nneg_ext
        if vote_neg_ext[i]<vote_pos_ext[i] and true_label == label1:
            ext_accuracy += 1./(Npos_ext+Nneg_ext)
            ext_sensitivity += 1./Npos_ext

    acc_ext_list.append(ext_accuracy)
    sens_ext_list.append(ext_sensitivity)
    spec_ext_list.append(ext_specificity)

    # Counting root feature in decision Trees
    
    for model in models:
        for estim in model.estimators_:
            feat_counter[estim.tree_.feature[0]] += 1
print('Done!                                  ')


prim_feat_trees = np.argsort(feat_counter)[::-1]
print("Most frequent argument in tree roots:")
print("F_num\tCount")
for i in range(10):
    print("{}\t{}".format(prim_feat_trees[i], feat_counter[prim_feat_trees[i]]))
print(np.sum(feat_counter))
print('\n')

base_acc_stat = acc_list
base_sens_stat = sens_list
base_spec_stat = spec_list

tot_acc = np.mean(acc_list)
tot_acc_std = np.std(acc_list)/np.sqrt(n_iter)
tot_sens = np.mean(sens_list)
tot_sens_std = np.std(sens_list)/np.sqrt(n_iter)
tot_spec = np.mean(spec_list)
tot_spec_std = np.std(spec_list)/np.sqrt(n_iter)

print("Performance of cross validation")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc, tot_acc_std, tot_acc_std/tot_acc))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens, tot_sens_std, tot_sens_std/tot_sens))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec, tot_spec_std, tot_spec_std/tot_spec))
print("\n")

base_acc_ext_stat = acc_ext_list
base_sens_ext_stat = sens_ext_list
base_spec_ext_stat = spec_ext_list

tot_acc_ext = np.mean(acc_ext_list)
tot_acc_ext_std = np.std(acc_ext_list)/np.sqrt(n_iter)
tot_sens_ext = np.mean(sens_ext_list)
tot_sens_ext_std = np.std(sens_ext_list)/np.sqrt(n_iter)
tot_spec_ext = np.mean(spec_ext_list)
tot_spec_ext_std = np.std(spec_ext_list)/np.sqrt(n_iter)

print("Performance of External test: Majority vote")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc_ext, tot_acc_ext_std, tot_acc_ext_std/tot_acc_ext))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens_ext, tot_sens_ext_std, tot_sens_ext_std/tot_sens_ext))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec_ext, tot_spec_ext_std, tot_spec_ext_std/tot_spec_ext))
#print("CPU Time:" + str((clock.process_time() - start_non_pca)))
print("\n")

tot_acc_bacc = np.mean(base_acc_bacc_list)
tot_acc_bacc_std = np.std(base_acc_bacc_list)/np.sqrt(n_iter)
tot_sens_bacc = np.mean(base_sens_bacc_list)
tot_sens_bacc_std = np.std(base_sens_bacc_list)/np.sqrt(n_iter)
tot_spec_bacc = np.mean(base_spec_bacc_list)
tot_spec_bacc_std = np.std(base_spec_bacc_list)/np.sqrt(n_iter)

print("Performance of External test: Best Accuracy in training")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc_bacc, tot_acc_bacc_std, tot_acc_bacc_std/tot_acc_bacc))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens_bacc, tot_sens_bacc_std, tot_sens_bacc_std/tot_sens_bacc))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec_bacc, tot_spec_bacc_std, tot_spec_bacc_std/tot_spec_bacc))
print("\n")

tot_acc_bsens = np.mean(base_acc_bsens_list)
tot_acc_bsens_std = np.std(base_acc_bsens_list)/np.sqrt(n_iter)
tot_sens_bsens = np.mean(base_sens_bsens_list)
tot_sens_bsens_std = np.std(base_sens_bsens_list)/np.sqrt(n_iter)
tot_spec_bsens = np.mean(base_spec_bsens_list)
tot_spec_bsens_std = np.std(base_spec_bsens_list)/np.sqrt(n_iter)

print("Performance of External test: Best Sensitivity in training")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc_bsens, tot_acc_bsens_std, tot_acc_bsens_std/tot_acc_bsens))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens_bsens, tot_sens_bsens_std, tot_sens_bsens_std/tot_sens_bsens))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec_bsens, tot_spec_bsens_std, tot_spec_bsens_std/tot_spec_bsens))
print("\n")

tot_acc_bspec = np.mean(base_acc_bspec_list)
tot_acc_bspec_std = np.std(base_acc_bspec_list)/np.sqrt(n_iter)
tot_sens_bspec = np.mean(base_sens_bspec_list)
tot_sens_bspec_std = np.std(base_sens_bspec_list)/np.sqrt(n_iter)
tot_spec_bspec = np.mean(base_spec_bspec_list)
tot_spec_bspec_std = np.std(base_spec_bspec_list)/np.sqrt(n_iter)

print("Performance of External test: Best Specificity in training")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc_bspec, tot_acc_bspec_std, tot_acc_bspec_std/tot_acc_bspec))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens_bspec, tot_sens_bspec_std, tot_sens_bspec_std/tot_sens_bspec))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec_bspec, tot_spec_bspec_std, tot_spec_bspec_std/tot_spec_bspec))
print("\n")

#PCA

pc_mat = pd.read_csv("eigenvectors.csv", sep = ',', index_col=0)

g0 = g0.dot(pc_mat)
g1 = g1.dot(pc_mat)
tot_pattern = np.concatenate((g0, g1), axis=0)

g0_ext = g0_ext.dot(pc_mat)
g1_ext = g1_ext.dot(pc_mat)
tot_ext_patt = np.concatenate((g0_ext, g1_ext), axis=0)

feat_counter = np.zeros(len(g0[0]))

acc_list = []
sens_list = []
spec_list = []

acc_ext_list = []
sens_ext_list = []
spec_ext_list = []

PCA_acc_bacc_list = []
PCA_sens_bacc_list = []
PCA_spec_bacc_list = []

PCA_acc_bsens_list = []
PCA_sens_bsens_list = []
PCA_spec_bsens_list = []

PCA_acc_bspec_list = []
PCA_sens_bspec_list = []
PCA_spec_bspec_list = []

print('Starting PCA iterations...')
#start_pca = clock.process_time()
for n in range(n_iter):
    # print("Iteration: {}/{} \r".format(n+1, n_iter)),
    print(f'Iteration: {n+1}/{n_iter}', end='\r' )

    models = []

    # Metrics
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []

    kf = KFold(n_splits = k_PCA, shuffle = True)
    indices = kf.split(tot_labels)

    vote_pos_ext = np.zeros(len(ext_labels))
    vote_neg_ext = np.zeros(len(ext_labels))

    for i, (train_index, test_index) in enumerate(indices):

        train_pattern = tot_pattern[train_index]
        train_labels = tot_labels[train_index]

        test_pattern = tot_pattern[test_index]
        test_labels = tot_labels[test_index]

        model = ens.RandomForestClassifier(n_estimators=n_trees_PCA, 
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

    acc_list.append(np.mean(fold_accuracy))
    sens_list.append(np.mean(fold_sensitivity))
    spec_list.append(np.mean(fold_specificity))

    # Performance of External Test: Best in Training

    bacc_model = models[np.argmax(fold_accuracy)]
    bacc_accuracy = 0.
    bacc_sensitivity = 0.
    bacc_specificity = 0.
    bacc_predict = bacc_model.predict(tot_ext_patt)

    for j, pred in enumerate(bacc_predict):
        if pred == label1 and ext_labels[j] == label1:
            bacc_accuracy += 1./(Npos_ext+Nneg_ext)
            bacc_specificity += 1./Nneg_ext
    
        if pred == label0 and ext_labels[j] == label0:
            bacc_accuracy += 1./(Npos_ext+Nneg_ext)
            bacc_sensitivity += 1./Npos_ext

    PCA_acc_bacc_list.append(bacc_accuracy)
    PCA_sens_bacc_list.append(bacc_sensitivity)
    PCA_spec_bacc_list.append(bacc_specificity)


    bsens_model = models[np.argmax(fold_sensitivity)]
    bsens_accuracy = 0.
    bsens_sensitivity = 0.
    bsens_specificity = 0.
    bsens_predict = bsens_model.predict(tot_ext_patt)

    for j, pred in enumerate(bsens_predict):
        if pred == label1 and ext_labels[j] == label1:
            bsens_accuracy += 1./(Npos_ext+Nneg_ext)
            bsens_specificity += 1./Nneg_ext
    
        if pred == label0 and ext_labels[j] == label0:
            bsens_accuracy += 1./(Npos_ext+Nneg_ext)
            bsens_sensitivity += 1./Npos_ext

    PCA_acc_bsens_list.append(bsens_accuracy)
    PCA_sens_bsens_list.append(bsens_sensitivity)
    PCA_spec_bsens_list.append(bsens_specificity)


    bspec_model = models[np.argmax(fold_specificity)]
    bspec_accuracy = 0.
    bspec_sensitivity = 0.
    bspec_specificity = 0.
    bspec_predict = bspec_model.predict(tot_ext_patt)

    for j, pred in enumerate(bspec_predict):
        if pred == label1 and ext_labels[j] == label1:
            bspec_accuracy += 1./(Npos_ext+Nneg_ext)
            bspec_specificity += 1./Nneg_ext
    
        if pred == label0 and ext_labels[j] == label0:
            bspec_accuracy += 1./(Npos_ext+Nneg_ext)
            bspec_sensitivity += 1./Npos_ext

    PCA_acc_bspec_list.append(bspec_accuracy)
    PCA_sens_bspec_list.append(bspec_sensitivity)
    PCA_spec_bspec_list.append(bspec_specificity)


    # Performance of External Test: Majority Vote

    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.


    for i, true_label in enumerate(ext_labels):
        if vote_neg_ext[i]>=vote_pos_ext[i] and true_label == label0:
            ext_accuracy += 1./(Npos_ext+Nneg_ext)
            ext_specificity += 1./Nneg_ext
        if vote_neg_ext[i]<vote_pos_ext[i] and true_label == label1:
            ext_accuracy += 1./(Npos_ext+Nneg_ext)
            ext_sensitivity += 1./Npos_ext

    acc_ext_list.append(ext_accuracy)
    sens_ext_list.append(ext_sensitivity)
    spec_ext_list.append(ext_specificity)

    # Counting root feature in decision Trees
    
    for model in models:
        for estim in model.estimators_:
            feat_counter[estim.tree_.feature[0]] += 1
print('Done!                                  ')


prim_feat_trees = np.argsort(feat_counter)[::-1]
print("Most frequent argument in tree roots:")
print("F_num\tCount")
for i in range(10):
    print("{}\t{}".format(prim_feat_trees[i], feat_counter[prim_feat_trees[i]]))
print(np.sum(feat_counter))
print('\n')

PCA_acc_stat = acc_list
PCA_sens_stat = sens_list
PCA_spec_stat = spec_list

PCA_acc = np.mean(acc_list)
PCA_acc_std = np.std(acc_list)/np.sqrt(n_iter)
PCA_sens = np.mean(sens_list)
PCA_sens_std = np.std(sens_list)/np.sqrt(n_iter)
PCA_spec = np.mean(spec_list)
PCA_spec_std = np.std(spec_list)/np.sqrt(n_iter)

print("Performance of cross validation")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_acc, PCA_acc_std, PCA_acc_std/PCA_acc))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_sens, PCA_sens_std, PCA_sens_std/PCA_sens))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_spec, PCA_spec_std, PCA_spec_std/PCA_spec))
print("\n")

PCA_acc_ext_stat = acc_ext_list
PCA_sens_ext_stat = sens_ext_list
PCA_spec_ext_stat = spec_ext_list

PCA_acc_ext = np.mean(acc_ext_list)
PCA_acc_ext_std = np.std(acc_ext_list)/np.sqrt(n_iter)
PCA_sens_ext = np.mean(sens_ext_list)
PCA_sens_ext_std = np.std(sens_ext_list)/np.sqrt(n_iter)
PCA_spec_ext = np.mean(spec_ext_list)
PCA_spec_ext_std = np.std(spec_ext_list)/np.sqrt(n_iter)

print("Performance of External test: Majority vote")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_acc_ext, PCA_acc_ext_std, PCA_acc_ext_std/PCA_acc_ext))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_sens_ext, PCA_sens_ext_std, PCA_sens_ext_std/PCA_sens_ext))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_spec_ext, PCA_spec_ext_std, PCA_spec_ext_std/PCA_spec_ext))
#print("CPU Time:" + str((clock.process_time() - start_pca)))
print("\n")

PCA_acc_bacc = np.mean(PCA_acc_bacc_list)
PCA_acc_bacc_std = np.std(PCA_acc_bacc_list)/np.sqrt(n_iter)
PCA_sens_bacc = np.mean(PCA_sens_bacc_list)
PCA_sens_bacc_std = np.std(PCA_sens_bacc_list)/np.sqrt(n_iter)
PCA_spec_bacc = np.mean(PCA_spec_bacc_list)
PCA_spec_bacc_std = np.std(PCA_spec_bacc_list)/np.sqrt(n_iter)

print("Performance of External test: Best Accuracy in training")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_acc_bacc, PCA_acc_bacc_std, PCA_acc_bacc_std/PCA_acc_bacc))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_sens_bacc, PCA_sens_bacc_std, PCA_sens_bacc_std/PCA_sens_bacc))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_spec_bacc, PCA_spec_bacc_std, PCA_spec_bacc_std/PCA_spec_bacc))
print("\n")

PCA_acc_bsens = np.mean(PCA_acc_bsens_list)
PCA_acc_bsens_std = np.std(PCA_acc_bsens_list)/np.sqrt(n_iter)
PCA_sens_bsens = np.mean(PCA_sens_bsens_list)
PCA_sens_bsens_std = np.std(PCA_sens_bsens_list)/np.sqrt(n_iter)
PCA_spec_bsens = np.mean(PCA_spec_bsens_list)
PCA_spec_bsens_std = np.std(PCA_spec_bsens_list)/np.sqrt(n_iter)

print("Performance of External test: Best Sensitivity in training")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_acc_bsens, PCA_acc_bsens_std, PCA_acc_bsens_std/PCA_acc_bsens))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_sens_bsens, PCA_sens_bsens_std, PCA_sens_bsens_std/PCA_sens_bsens))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_spec_bsens, PCA_spec_bsens_std, PCA_spec_bsens_std/PCA_spec_bsens))
print("\n")

PCA_acc_bspec = np.mean(PCA_acc_bspec_list)
PCA_acc_bspec_std = np.std(PCA_acc_bspec_list)/np.sqrt(n_iter)
PCA_sens_bspec = np.mean(PCA_sens_bspec_list)
PCA_sens_bspec_std = np.std(PCA_sens_bspec_list)/np.sqrt(n_iter)
PCA_spec_bspec = np.mean(PCA_spec_bspec_list)
PCA_spec_bspec_std = np.std(PCA_spec_bspec_list)/np.sqrt(n_iter)

print("Performance of External test: Best Specificity in training")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_acc_bspec, PCA_acc_bspec_std, PCA_acc_bspec_std/PCA_acc_bspec))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_sens_bspec, PCA_sens_bspec_std, PCA_sens_bspec_std/PCA_sens_bspec))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_spec_bspec, PCA_spec_bspec_std, PCA_spec_bspec_std/PCA_spec_bspec))
print("\n")



print('Cross Validation Statistics:')
base_acc_res = ttest_ind(base_acc_stat, PCA_acc_stat, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(base_acc_res.statistic, base_acc_res.pvalue))
base_sens_res = ttest_ind(base_sens_stat, PCA_sens_stat, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(base_sens_res.statistic, base_sens_res.pvalue))
base_spec_res = ttest_ind(base_spec_stat, PCA_spec_stat, equal_var=False)
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(base_spec_res.statistic, base_spec_res.pvalue))

print('External Test (Majority vote) Statistics:')
ext_acc_res = ttest_ind(base_acc_ext_stat, PCA_acc_ext_stat, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(ext_acc_res.statistic, ext_acc_res.pvalue))
ext_sens_res = ttest_ind(base_sens_ext_stat, PCA_sens_ext_stat, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(ext_sens_res.statistic, ext_sens_res.pvalue))
ext_spec_res = ttest_ind(base_spec_ext_stat, PCA_spec_ext_stat, equal_var=False)
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(ext_spec_res.statistic, ext_spec_res.pvalue))

print('External Test (Best Accuracy in Training) Statistics:')
bacc_acc_res = ttest_ind(base_acc_bacc_list, PCA_acc_bacc_list, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(bacc_acc_res.statistic, bacc_acc_res.pvalue))
bacc_sens_res = ttest_ind(base_sens_bacc_list, PCA_sens_bacc_list, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(bacc_sens_res.statistic, bacc_sens_res.pvalue))
bacc_spec_res = ttest_ind(base_spec_bacc_list, PCA_spec_bacc_list, equal_var=False)
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(bacc_spec_res.statistic, bacc_spec_res.pvalue))

print('External Test (Best Sensitivity in Training) Statistics:')
bsens_acc_res = ttest_ind(base_acc_bsens_list, PCA_acc_bsens_list, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(bsens_acc_res.statistic, bsens_acc_res.pvalue))
bsens_sens_res = ttest_ind(base_sens_bsens_list, PCA_sens_bsens_list, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(bsens_sens_res.statistic, bsens_sens_res.pvalue))
bsens_spec_res = ttest_ind(base_spec_bsens_list, PCA_spec_bsens_list, equal_var=False)
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(bsens_spec_res.statistic, bsens_spec_res.pvalue))

print('External Test (Best Specificity in Training) Statistics:')
bspec_acc_res = ttest_ind(base_acc_bspec_list, PCA_acc_bspec_list, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(bspec_acc_res.statistic, bspec_acc_res.pvalue))
bspec_sens_res = ttest_ind(base_sens_bspec_list, PCA_sens_bspec_list, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(bspec_sens_res.statistic, bspec_sens_res.pvalue))
bspec_spec_res = ttest_ind(base_spec_bspec_list, PCA_spec_bspec_list, equal_var=False)
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(bspec_spec_res.statistic, bspec_spec_res.pvalue))

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")

dict = {'Base CV Avg': [tot_acc, tot_sens, tot_spec],
        'Base CV Std': [tot_acc_std, tot_sens_std, tot_spec_std],
        'Base Ext Avg': [tot_acc_ext, tot_sens_ext, tot_spec_ext],
        'Base Ext Std': [tot_acc_ext_std, tot_sens_ext_std, tot_spec_ext_std],
        'Base BAcc Avg': [tot_acc_bacc, tot_sens_bacc, tot_spec_bacc],
        'Base BAcc Std': [tot_acc_bacc_std, tot_sens_bacc_std, tot_spec_bacc_std],
        'Base BSens Avg': [tot_acc_bsens, tot_sens_bsens, tot_spec_bsens],
        'Base BSens Std': [tot_acc_bsens_std, tot_sens_bsens_std, tot_spec_bsens_std],
        'Base BSpec Avg': [tot_acc_bspec, tot_sens_bspec, tot_spec_bspec],
        'Base BSpec Std': [tot_acc_bspec_std, tot_sens_bspec_std, tot_spec_bspec_std],
        'PCA CV Avg': [PCA_acc, PCA_sens, PCA_spec],
        'PCA CV Std': [PCA_acc_std, PCA_sens_std, PCA_spec_std],
        'PCA Ext Avg': [PCA_acc_ext, PCA_sens_ext, PCA_spec_ext],
        'PCA BAcc Avg': [PCA_acc_bacc, PCA_sens_bacc, PCA_spec_bacc],
        'PCA BAcc Std': [PCA_acc_bacc_std, PCA_sens_bacc_std, PCA_spec_bacc_std],
        'PCA BSens Avg': [PCA_acc_bsens, PCA_sens_bsens, PCA_spec_bsens],
        'PCA BSens Std': [PCA_acc_bsens_std, PCA_sens_bsens_std, PCA_spec_bsens_std],
        'PCA BSpec Avg': [PCA_acc_bspec, PCA_sens_bspec, PCA_spec_bspec],
        'PCA BSpec Std': [PCA_acc_bspec_std, PCA_sens_bspec_std, PCA_spec_bspec_std],
        'PCA Ext Std': [PCA_acc_ext_std, PCA_sens_ext_std, PCA_spec_ext_std],
        'CV t-test': [base_acc_res.statistic, base_sens_res.statistic, base_spec_res.statistic],
        'CV p-value': [base_acc_res.pvalue, base_sens_res.pvalue, base_spec_res.pvalue],
        'Ext t-test': [ext_acc_res.statistic, ext_sens_res.statistic, ext_spec_res.statistic],
        'Ext p-value': [ext_acc_res.pvalue, ext_sens_res.pvalue, ext_spec_res.pvalue],
        'BAcc t-test': [bacc_acc_res.statistic, bacc_sens_res.statistic, bacc_spec_res.statistic],
        'BAcc p-value': [bacc_acc_res.pvalue, bacc_sens_res.pvalue, bacc_spec_res.pvalue],
        'BSens t-test': [bsens_acc_res.statistic, bsens_sens_res.statistic, bsens_spec_res.statistic],
        'BSens p-value': [bsens_acc_res.pvalue, bsens_sens_res.pvalue, bsens_spec_res.pvalue],
        'BSpec t-test': [bspec_acc_res.statistic, bspec_sens_res.statistic, bspec_spec_res.statistic],
        'BSpec p-value': [bspec_acc_res.pvalue, bspec_sens_res.pvalue, bspec_spec_res.pvalue]}

results = pd.DataFrame(data = dict, index = ['Accuracy', 'Sensitivity', 'Specificity'])
results.to_csv('results.csv')