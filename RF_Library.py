import numpy as np
import sklearn.ensemble as ens
import multiprocessing as mp
from sklearn.model_selection import KFold


def RF_binary_kfold(n_trees, k, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []

    kf = KFold(n_splits = k, shuffle = True)
    indices = kf.split(labels)

    if ext_lab != None:
        vote_1_ext = np.zeros(len(ext_lab))
        vote_0_ext = np.zeros(len(ext_lab))

        N1_ext = 0
        N0_ext = 0
        for lab in ext_lab:
            if lab == label0:
                N0_ext +=1
            if lab == label1:
                N1_ext +=1

    for i, (train_index, test_index) in enumerate(indices):

        train_pattern = patterns[train_index]
        train_labels = labels[train_index]

        test_pattern = patterns[test_index]
        test_labels = labels[test_index]



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

        if ext_lab != None:
            ext_pred = model.predict(ext_patt)

            for j, pred in enumerate(ext_pred):
                if pred == label0:
                    vote_0_ext[j] += 1
                if pred == label1:
                    vote_1_ext[j] += 1

    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.

    if ext_lab != None:
        for i, true_label in enumerate(ext_lab):
            if vote_0_ext[i]>=vote_1_ext[i] and true_label == label0:
                ext_accuracy += 1./(N1_ext+N0_ext)
                ext_specificity += 1./N0_ext
            if vote_0_ext[i]<vote_1_ext[i] and true_label == label1:
                ext_accuracy += 1./(N1_ext+N0_ext)
                ext_sensitivity += 1./N1_ext

    res = {
        'n trees': n_trees,
        'k': k,
        'Acc': np.mean(fold_accuracy),
        'Acc Err': np.std(fold_accuracy)/np.sqrt(k),
        'Sens': np.mean(fold_sensitivity),
        'Sens Err': np.std(fold_sensitivity)/np.sqrt(k),
        'Spec': np.mean(fold_specificity),
        'Spec Err': np.std(fold_specificity)/np.sqrt(k),
        'Ext Acc': ext_accuracy,
        'Ext Sens': ext_sensitivity,
        'Ext Spec': ext_specificity
    }

    return res

def RF_binary_scanner(tree_range, k_range, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    len_tree = len(tree_range)
    len_k = len(k_range)

    accuracy_list = np.empty((len_tree, len_k))
    accuracy_err_list = np.empty((len_tree, len_k))
    sensitivity_list = np.empty((len_tree, len_k))
    sensitivity_err_list = np.empty((len_tree, len_k))
    specificity_list = np.empty((len_tree, len_k))
    specificity_err_list = np.empty((len_tree, len_k))

    print("Begin Scanning...")

    tot_iter = len_tree*len_k
    iter = 0

    for i_trees, n_trees in enumerate(tree_range):
        for i_k, k in enumerate(k_range):
            iter += 1
            print("N_trees: ", n_trees, "\tN_fold: ", k, "\tIteration: ", iter, "/", tot_iter,end='\r')     #Python 3.x
            # print("N_trees: {}\tN_fold: {}\tIteration: {}/{} \r".format(n_trees, k, iter, tot_iter)),       #Python 2.x

            res = RF_binary_kfold(n_trees, k, patterns, labels, label0, label1)

            accuracy_list[i_trees, i_k] = res['Acc']
            accuracy_err_list[i_trees, i_k] = res['Acc Err']
            sensitivity_list[i_trees, i_k] = res['Sens']
            sensitivity_err_list[i_trees, i_k] = res['Sens Err']
            specificity_list[i_trees, i_k] = res['Spec']
            specificity_err_list[i_trees, i_k] = res['Spec Err']

    print("Finished Scanning!                                                \n")

    return accuracy_list, accuracy_err_list, sensitivity_list, sensitivity_err_list, specificity_list, specificity_err_list




