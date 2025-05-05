import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import time as clock
import pandas as pd
import tkinter as tk
import os

from scipy.stats import ttest_ind
from sklearn.model_selection import KFold
from scipy.io import loadmat
from scipy.optimize import curve_fit

import RF_Library as RF     # Custom made functions

def h_line(x,c):
    return c

start = clock.time()

n_iter = 100

# Iperparameters non-PCA
k = 6
n_trees = 200

# Iperparameters PCA
k_PCA = 6
n_trees_PCA = 250


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
ext_patt = np.concatenate((g0_ext, g1_ext), axis=0)

Nneg_ext = len(g0_ext)
Npos_ext = len(g1_ext)

g0_ext_labels = np.full(Nneg_ext, label1)                           #Labels are inverted!
g1_ext_labels = np.full(Npos_ext, label0)
ext_labels = np.concatenate((g0_ext_labels, g1_ext_labels), axis=0)

feat_counter = np.zeros(len(g0[0]))

acc_list = []                   #CV metrics
sens_list = []
spec_list = []

acc_ext_list = []               #Majority vote
sens_ext_list = []
spec_ext_list = []
conf_mat_list = []

base_acc_bacc_list = []         #Best Accuracy in training
base_sens_bacc_list = []
base_spec_bacc_list = []

base_acc_bsens_list = []        #Best Sensitivity in training
base_sens_bsens_list = []
base_spec_bsens_list = []

base_acc_bspec_list = []        #Best Specificity in training
base_sens_bspec_list = []
base_spec_bspec_list = []

print('Starting Non-PCA iterations...')
start_non_pca = clock.process_time()                               #CPU time is different in 2.7-
cpu_time_non_pca = []
for n in range(n_iter):
    start_single_non_pca= clock.process_time()  
    # print("Iteration: {}/{} \r".format(n+1, n_iter)),
    print(f'Iteration: {n+1}/{n_iter}', end='\r' )

    res = RF.RF_binary_kfold(n_trees, k, tot_pattern, tot_labels, label0, label1, ext_patt, ext_labels)

    acc_list.append(res['Acc'])
    sens_list.append(res['Sens'])
    spec_list.append(res['Spec'])

    base_acc_bacc_list.append(res['BAcc Acc'])
    base_sens_bacc_list.append(res['BAcc Sens'])
    base_spec_bacc_list.append(res['BAcc Spec'])

    base_acc_bsens_list.append(res['BSens Acc'])
    base_sens_bsens_list.append(res['BSens Sens'])
    base_spec_bsens_list.append(res['BSens Spec'])

    base_acc_bspec_list.append(res['BSpec Acc'])
    base_sens_bspec_list.append(res['BSpec Sens'])
    base_spec_bspec_list.append(res['BSpec Spec'])

    acc_ext_list.append(res['Ext Acc'])
    sens_ext_list.append(res['Ext Sens'])
    spec_ext_list.append(res['Ext Spec'])
    conf_mat_list.append(res['Conf Mat'])

    # Counting root feature in decision Trees
    for model in res['Models']:
        for estim in model.estimators_:
            feat_counter[estim.tree_.feature[0]] += 1

    cpu_time_stamp = clock.process_time() - start_single_non_pca
    cpu_time_non_pca.append(cpu_time_stamp)
    print(f'\t\t\tCPU Time: {cpu_time_stamp}', end='\r')
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
tot_conf_mat = np.empty((2,2))
tot_conf_mat = np.mean(conf_mat_list, axis=0)


print("Performance of External test: Majority vote")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc_ext, tot_acc_ext_std, tot_acc_ext_std/tot_acc_ext))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens_ext, tot_sens_ext_std, tot_sens_ext_std/tot_sens_ext))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec_ext, tot_spec_ext_std, tot_spec_ext_std/tot_spec_ext))
print("Mean CPU Time for single iteration: {:.2}".format(np.mean(cpu_time_non_pca)))
print("Tot CPU Time:" + str((clock.process_time() - start_non_pca)))
print("\n")

fig_baseCM, ax_baseCM = RF.confMat_binary_plot(tot_conf_mat, title="Confusion Matrix - non PCA")
fig_baseCM.savefig("nonPCA Confusion Matrix.pdf")

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
ext_patt = np.concatenate((g0_ext, g1_ext), axis=0)


feat_counter = np.zeros(len(g0[0]))

acc_list = []                   #CV metrics
sens_list = []
spec_list = []

acc_ext_list = []               #Majority vote
sens_ext_list = []
spec_ext_list = []
conf_mat_list = []

PCA_acc_bacc_list = []          #Best Accuracy in training
PCA_sens_bacc_list = []
PCA_spec_bacc_list = []

PCA_acc_bsens_list = []         #Best sensitivity in Training
PCA_sens_bsens_list = []
PCA_spec_bsens_list = []

PCA_acc_bspec_list = []         #Best Specificity in training
PCA_sens_bspec_list = []
PCA_spec_bspec_list = []

print('Starting PCA iterations...')
start_pca = clock.process_time()
cpu_time_pca = []
for n in range(n_iter):
    start_single_pca= clock.process_time() 
    # print("Iteration: {}/{} \r".format(n+1, n_iter)),
    print(f'Iteration: {n+1}/{n_iter}', end='\r' )

    res = RF.RF_binary_kfold(n_trees_PCA, k_PCA, tot_pattern, tot_labels, label0, label1, ext_patt, ext_labels)

    acc_list.append(res['Acc'])
    sens_list.append(res['Sens'])
    spec_list.append(res['Spec'])

    PCA_acc_bacc_list.append(res['BAcc Acc'])
    PCA_sens_bacc_list.append(res['BAcc Sens'])
    PCA_spec_bacc_list.append(res['BAcc Spec'])

    PCA_acc_bsens_list.append(res['BSens Acc'])
    PCA_sens_bsens_list.append(res['BSens Sens'])
    PCA_spec_bsens_list.append(res['BSens Spec'])

    PCA_acc_bspec_list.append(res['BSpec Acc'])
    PCA_sens_bspec_list.append(res['BSpec Sens'])
    PCA_spec_bspec_list.append(res['BSpec Spec'])

    acc_ext_list.append(res['Ext Acc'])
    sens_ext_list.append(res['Ext Sens'])
    spec_ext_list.append(res['Ext Spec'])
    conf_mat_list.append(res['Conf Mat'])

    # Counting root feature in decision Trees
    for model in res['Models']:
        for estim in model.estimators_:
            feat_counter[estim.tree_.feature[0]] += 1

    cpu_time_stamp = clock.process_time() - start_single_pca
    cpu_time_pca.append(cpu_time_stamp)
    print(f'\t\t\tCPU Time: {cpu_time_stamp}', end='\r')
print('Done!                                                ')


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
PCA_conf_mat = np.empty((2,2))
PCA_conf_mat = np.mean(conf_mat_list, axis=0)

print("Performance of External test: Majority vote")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_acc_ext, PCA_acc_ext_std, PCA_acc_ext_std/PCA_acc_ext))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_sens_ext, PCA_sens_ext_std, PCA_sens_ext_std/PCA_sens_ext))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(PCA_spec_ext, PCA_spec_ext_std, PCA_spec_ext_std/PCA_spec_ext))
print("CPU Time:" + str((clock.process_time() - start_pca)) + ' s')
print("\n")

fig_PCACM, ax_PCACM = RF.confMat_binary_plot(PCA_conf_mat, title="Confusion matrix - PCA")
fig_PCACM.savefig("PCA Confusion Matrix.pdf")

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


pca_label = 'PCA'
not_pca_label = 'NON PCA'

if not os.path.exists('distributions/cross_validation'):
        os.makedirs('distributions/cross_validation')
if not os.path.exists('distributions/cross_validation/majority'):
        os.makedirs('distributions/cross_validation/majority')

if not os.path.exists('distributions/external_test'):
        os.makedirs('distributions/external_test')
if not os.path.exists('distributions/external_test/majority'):
        os.makedirs('distributions/external_test/majority')
if not os.path.exists('distributions/external_test/best_model'):
        os.makedirs('distributions/external_test/best_model')  
if not os.path.exists('distributions/external_test/best_model/best_accuracy'):
        os.makedirs('distributions/external_test/best_model/best_accuracy')  
if not os.path.exists('distributions/external_test/best_model/best_sensitivity'):
        os.makedirs('distributions/external_test/best_model/best_sensitivity')  
if not os.path.exists('distributions/external_test/best_model/best_specificity'):
        os.makedirs('distributions/external_test/best_model/best_specificity')  

print('Cross Validation Statistics:')
cv_maj_acc_fig, cv_maj_acc_ax, base_acc_res = RF.plot_histo_gaus_stat(base_acc_stat, not_pca_label, PCA_acc_stat, pca_label)
cv_maj_acc_ax.set_title('Accuracy Distributions Cross Validation Majority Vote')
cv_maj_acc_fig.savefig('distributions/cross_validation/majority/accuracy_dist_maj_cv.png')
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(base_acc_res.statistic, base_acc_res.pvalue))
cv_maj_sens_fig, cv_maj_sens_ax, base_sens_res = RF.plot_histo_gaus_stat(base_sens_stat, not_pca_label, PCA_sens_stat, pca_label)
cv_maj_sens_ax.set_title('Sensitivity Distribution Cross Validation Majority Vote')
cv_maj_sens_fig.savefig('distributions/cross_validation/majority/sensitivity_dist_maj_cv.png')
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(base_sens_res.statistic, base_sens_res.pvalue))
cv_maj_spec_fig, cv_maj_spec_ax, base_spec_res = RF.plot_histo_gaus_stat(base_spec_stat, not_pca_label, PCA_spec_stat, pca_label)
cv_maj_spec_ax.set_title('Specificity Distribution Cross Validation Majority Vote')
cv_maj_spec_fig.savefig('distributions/cross_validation/majority/specificity_dist_maj_cv.png')
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(base_spec_res.statistic, base_spec_res.pvalue))

print('External Test (Majority vote) Statistics:')
ext_maj_acc_fig, ext_maj_acc_ax, ext_acc_res = RF.plot_histo_gaus_stat(base_acc_ext_stat, not_pca_label, PCA_acc_ext_stat, pca_label)
ext_maj_acc_ax.set_title('Accuracy Distribution External Test Majority Vote')
ext_maj_acc_fig.savefig('distributions/external_test/majority/accuracy_dist_maj_ext.png')
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(ext_acc_res.statistic, ext_acc_res.pvalue))
ext_maj_sens_fig, ext_maj_sens_ax, ext_sens_res = RF.plot_histo_gaus_stat(base_sens_ext_stat, not_pca_label, PCA_sens_ext_stat, pca_label)
ext_maj_sens_ax.set_title('Sensitivity Distribution External Test Majority Vote')
ext_maj_sens_fig.savefig('distributions/external_test/majority/sensitivity_dist_maj_ext.png')
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(ext_sens_res.statistic, ext_sens_res.pvalue))
ext_maj_spec_fig, ext_maj_spec_ax, ext_spec_res = RF.plot_histo_gaus_stat(base_spec_ext_stat, not_pca_label, PCA_spec_ext_stat, pca_label)
ext_maj_spec_ax.set_title('Specificity Distribution External Test Majority Vote')
ext_maj_spec_fig.savefig('distributions/external_test/majority/specificity_dist_maj_ext.png')
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(ext_spec_res.statistic, ext_spec_res.pvalue))

print('External Test (Best Accuracy in Training) Statistics:')
ext_bacc_acc_fig, ext_bacc_acc_ax, bacc_acc_res = RF.plot_histo_gaus_stat(base_acc_bacc_list, not_pca_label, PCA_acc_bacc_list, pca_label)
ext_bacc_acc_ax.set_title('Accuracy Distribution External Test Best Accuracy in Training')
ext_bacc_acc_fig.savefig('distributions/external_test/best_model/best_accuracy/accuracy_dist_bacc_ext.png')
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(bacc_acc_res.statistic, bacc_acc_res.pvalue))
ext_bacc_sens_fig, ext_bacc_sens_ax, bacc_sens_res = RF.plot_histo_gaus_stat(base_sens_bacc_list, not_pca_label, PCA_sens_bacc_list, pca_label)
ext_bacc_sens_ax.set_title('Sensitivity Distribution External Test Best Accuracy in Training')
ext_bacc_sens_fig.savefig('distributions/external_test/best_model/best_accuracy/sensitivity_dist_bacc_ext.png')
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(bacc_sens_res.statistic, bacc_sens_res.pvalue))
ext_bacc_spec_fig, ext_bacc_spec_ax, bacc_spec_res = RF.plot_histo_gaus_stat(base_spec_bacc_list, not_pca_label, PCA_spec_bacc_list, pca_label)
ext_bacc_spec_ax.set_title('Specificity Distribution External Test Best Accuracy in Training')
ext_bacc_spec_fig.savefig('distributions/external_test/best_model/best_accuracy/specificity_dist_bacc_ext.png')
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(bacc_spec_res.statistic, bacc_spec_res.pvalue))

print('External Test (Best Sensitivity in Training) Statistics:')
ext_bsens_acc_fig, ext_bsens_acc_ax, bsens_acc_res = RF.plot_histo_gaus_stat(base_acc_bsens_list, not_pca_label, PCA_acc_bsens_list, pca_label)
ext_bsens_acc_ax.set_title('Accuracy Distribution External Test Best Sensitivity in Training')
ext_bsens_acc_fig.savefig('distributions/external_test/best_model/best_sensitivity/accuracy_dist_bsens_ext.png')
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(bsens_acc_res.statistic, bsens_acc_res.pvalue))
ext_bsens_sens_fig, ext_bsens_sens_ax, bsens_sens_res = RF.plot_histo_gaus_stat(base_sens_bsens_list, not_pca_label, PCA_sens_bsens_list, pca_label)
ext_bsens_sens_ax.set_title('Sensitivity Distribution External Test Best Sensitivity in Training')
ext_bsens_sens_fig.savefig('distributions/external_test/best_model/best_sensitivity/sensitivity_dist_bsens_ext.png')
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(bsens_sens_res.statistic, bsens_sens_res.pvalue))
ext_bsens_spec_fig, ext_bsens_spec_ax, bsens_spec_res = RF.plot_histo_gaus_stat(base_spec_bsens_list, not_pca_label, PCA_spec_bsens_list, pca_label)
ext_bsens_spec_ax.set_title('Specificity Distribution External Test Best Sensitivity in Training')
ext_bsens_spec_fig.savefig('distributions/external_test/best_model/best_sensitivity/specificity_dist_bsens_ext.png')
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(bsens_spec_res.statistic, bsens_spec_res.pvalue))

print('External Test (Best Specificity in Training) Statistics:')
ext_bspec_acc_fig, ext_bspec_acc_ax, bspec_acc_res = RF.plot_histo_gaus_stat(base_acc_bspec_list, not_pca_label, PCA_acc_bspec_list, pca_label)
ext_bspec_acc_ax.set_title('Accuracy Distribution External Test Best Specificity in Training')
ext_bspec_acc_fig.savefig('distributions/external_test/best_model/best_specificity/accuracy_dist_bspec_ext.png')
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(bspec_acc_res.statistic, bspec_acc_res.pvalue))
ext_bspec_sens_fig, ext_bspec_sens_ax, bspec_sens_res = RF.plot_histo_gaus_stat(base_sens_bspec_list, not_pca_label, PCA_sens_bspec_list, pca_label)
ext_bspec_sens_ax.set_title('Sensitivity Distribution External Test Best Specificity in Training')
ext_bspec_sens_fig.savefig('distributions/external_test/best_model/best_specificity/sensitivity_dist_bspec_ext.png')
print('Sensitivity t-stat: {:.2}, p-value: {:.2}'.format(bspec_sens_res.statistic, bspec_sens_res.pvalue))
ext_bspec_spec_fig, ext_bspec_spec_ax, bspec_spec_res = RF.plot_histo_gaus_stat(base_spec_bspec_list, not_pca_label, PCA_spec_bspec_list, pca_label)
ext_bspec_spec_ax.set_title('Specificity Distribution External Test Best Specificity in Training')
ext_bspec_spec_fig.savefig('distributions/external_test/best_model/best_specificity/specificity_dist_bspec_ext.png')
print('Specificity t-stat: {:.2}, p-value: {:.2}\n'.format(bspec_spec_res.statistic, bspec_spec_res.pvalue))

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")

dict = {'Base CV Avg': [tot_acc, tot_sens, tot_spec],
        'Base CV Std': [tot_acc_std, tot_sens_std, tot_spec_std],
        'PCA CV Avg': [PCA_acc, PCA_sens, PCA_spec],
        'PCA CV Std': [PCA_acc_std, PCA_sens_std, PCA_spec_std],
        'CV t-test': [base_acc_res.statistic, base_sens_res.statistic, base_spec_res.statistic],
        'CV p-value': [base_acc_res.pvalue, base_sens_res.pvalue, base_spec_res.pvalue],

        'Base Ext Avg': [tot_acc_ext, tot_sens_ext, tot_spec_ext],
        'Base Ext Std': [tot_acc_ext_std, tot_sens_ext_std, tot_spec_ext_std],
        'PCA Ext Avg': [PCA_acc_ext, PCA_sens_ext, PCA_spec_ext],
        'PCA Ext Std': [PCA_acc_ext_std, PCA_sens_ext_std, PCA_spec_ext_std],
        'Ext t-test': [ext_acc_res.statistic, ext_sens_res.statistic, ext_spec_res.statistic],
        'Ext p-value': [ext_acc_res.pvalue, ext_sens_res.pvalue, ext_spec_res.pvalue],

        'Base BAcc Avg': [tot_acc_bacc, tot_sens_bacc, tot_spec_bacc],
        'Base BAcc Std': [tot_acc_bacc_std, tot_sens_bacc_std, tot_spec_bacc_std],
        'PCA BAcc Avg': [PCA_acc_bacc, PCA_sens_bacc, PCA_spec_bacc],
        'PCA BAcc Std': [PCA_acc_bacc_std, PCA_sens_bacc_std, PCA_spec_bacc_std],
        'BAcc t-test': [bacc_acc_res.statistic, bacc_sens_res.statistic, bacc_spec_res.statistic],
        'BAcc p-value': [bacc_acc_res.pvalue, bacc_sens_res.pvalue, bacc_spec_res.pvalue],

        'Base BSens Avg': [tot_acc_bsens, tot_sens_bsens, tot_spec_bsens],
        'Base BSens Std': [tot_acc_bsens_std, tot_sens_bsens_std, tot_spec_bsens_std],
        'PCA BSens Avg': [PCA_acc_bsens, PCA_sens_bsens, PCA_spec_bsens],
        'PCA BSens Std': [PCA_acc_bsens_std, PCA_sens_bsens_std, PCA_spec_bsens_std],
        'BSens t-test': [bsens_acc_res.statistic, bsens_sens_res.statistic, bsens_spec_res.statistic],
        'BSens p-value': [bsens_acc_res.pvalue, bsens_sens_res.pvalue, bsens_spec_res.pvalue],

        'Base BSpec Avg': [tot_acc_bspec, tot_sens_bspec, tot_spec_bspec],
        'Base BSpec Std': [tot_acc_bspec_std, tot_sens_bspec_std, tot_spec_bspec_std],
        'PCA BSpec Avg': [PCA_acc_bspec, PCA_sens_bspec, PCA_spec_bspec],
        'PCA BSpec Std': [PCA_acc_bspec_std, PCA_sens_bspec_std, PCA_spec_bspec_std],
        'BSpec t-test': [bspec_acc_res.statistic, bspec_sens_res.statistic, bspec_spec_res.statistic],
        'BSpec p-value': [bspec_acc_res.pvalue, bspec_sens_res.pvalue, bspec_spec_res.pvalue]}

results = pd.DataFrame(data = dict, index = ['Accuracy', 'Sensitivity', 'Specificity'])
results.to_csv('results.csv')

# root = tk.Tk()
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()
# root.destroy() 

# time_fig, time_ax = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
time_fig, time_ax = plt.subplots()
iterations = np.arange(0,n_iter,1)

popt_not_pca, pcov_not_pca = curve_fit(h_line, iterations,cpu_time_non_pca, maxfev=10000)
popt_pca, pcov_pca = curve_fit(h_line, iterations,cpu_time_pca, maxfev=10000)

time_ax.plot(iterations, cpu_time_non_pca, 'rx', label=f'NOT PCA:{popt_not_pca[0]:.2f} s')
time_ax.hlines(popt_not_pca[0], 0, n_iter, colors='red', linestyles='dashed')
time_ax.plot(iterations, cpu_time_pca, 'bo', label=f'PCA:{popt_pca[0]:.2f} s')
time_ax.hlines(popt_pca[0], 0, n_iter, colors='blue', linestyles='dashed')
time_ax.set_xlabel('Iteration')
time_ax.set_ylabel('CPU Time (s)')
time_ax.grid(True)
time_ax.legend(loc='best', fontsize='large')
time_ax.set_title('CPU Time')
time_fig.savefig('CPU_time_iterations.png')

plt.show()