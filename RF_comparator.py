import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import time as clock
import pandas as pd
import tkinter as tk 

from scipy.stats import ttest_ind
from sklearn.model_selection import KFold
from scipy.io import loadmat
from scipy.optimize import curve_fit

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

import RF_Library as RF     # Custom made functions

start = clock.time()

n_iter = 1000

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
for n in range(n_iter):
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
    for model in res['models']:
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
tot_conf_mat = np.empty((2,2))
tot_conf_mat = np.mean(conf_mat_list, axis=0)


print("Performance of External test: Majority vote")
print("Accuracy: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_acc_ext, tot_acc_ext_std, tot_acc_ext_std/tot_acc_ext))
print("Sensitivity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_sens_ext, tot_sens_ext_std, tot_sens_ext_std/tot_sens_ext))
print("Specificity: {:.2%} +- {:.2%} Rel: {:.2%}".format(tot_spec_ext, tot_spec_ext_std, tot_spec_ext_std/tot_spec_ext))
print("CPU Time:" + str((clock.process_time() - start_non_pca)) + ' s')
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
for n in range(n_iter):
    # print("Iteration: {}/{} \r".format(n+1, n_iter)),
    print(f'Iteration: {n+1}/{n_iter}', end='\r' )

    res = RF.RF_binary_kfold(n_trees_PCA, k_PCA, tot_pattern, tot_labels, label0, label1, ext_patt, ext_labels)

    acc_list.append(res['Acc'])
    sens_list.append(res['Sens'])
    spec_list.append(res['Spec'])

    PCA_acc_bacc_list.append(res['BAcc Acc'])
    PCA_sens_bacc_list.append(res['Bcc Sens'])
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
    for model in res['models']:
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


root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()  # Close the tkinter window

fig, ax = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
cv_sens_bin_vals, cv_sens_bins, _ = ax.hist(cross_sens_stat, bins='auto', alpha = 0.5, color='red', label ='NON PCA')
PCA_sens_bin_vals, PCA_sens_bins, _ = ax.hist(PCA_sens_stat, bins='auto', alpha = 0.5, color='blue', label ='PCA')

cv_sens_bin_centers = (cv_sens_bins[:-1] + cv_sens_bins[1:])/2
i_max = np.argmax(cv_sens_bin_vals)
par = [cv_sens_bin_vals[i_max], cv_sens_bin_centers[i_max], 0.05]
popt_cv_sens, pcov_cv_sens = curve_fit(gaussian, cv_sens_bin_centers, cv_sens_bin_vals, par, maxfev=10000)

PCA_sens_bin_centers = (PCA_sens_bins[:-1] + PCA_sens_bins[1:])/2
i_max = np.argmax(PCA_sens_bin_vals)
par = [PCA_sens_bin_vals[i_max], PCA_sens_bin_centers[i_max], 0.05]
popt_PCA_cv_sens, pcov_PCA_cv_sens = curve_fit(gaussian, PCA_sens_bin_centers, PCA_sens_bin_vals, par, maxfev=10000)

x = np.linspace(np.min(np.concatenate((cv_sens_bins, PCA_sens_bins))), np.max(np.concatenate((cv_sens_bins, PCA_sens_bins))), 1000)
ax.plot(x, gaussian(x,*popt_cv_sens), 'r--', label=f'Gaussian Fit: $\mu$ = {popt_cv_sens[1]:.2f}, $\sigma$ = {popt_cv_sens[2]:.2f}, A = {popt_cv_sens[0]:.2f}')
ax.plot(x, gaussian(x,*popt_PCA_cv_sens), 'b--', label=f'Gaussian Fit: $\mu$ = {popt_PCA_cv_sens[1]:.2f}, $\sigma$ = {popt_PCA_cv_sens[2]:.2f}, A = {popt_PCA_cv_sens[0]:.2f}')

cross_sens_res = ttest_ind(gaussian(x,*popt_cv_sens), gaussian(x,*popt_PCA_cv_sens), equal_var=False, alternative='greater')
ax.plot([],[], marker= None, linestyle='None', label=f'p-value fit: {cross_sens_res.pvalue:.2f}')

plt.legend()
plt.title('Sensitivity Distributions Cross Validation')
plt.savefig('sensitivity_dist_cross_validation.png')

fig1, ax1 = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
cv_spec_bin_vals, cv_spec_bins, _ = ax1.hist(cross_spec_stat, bins='auto', alpha = 0.5, color='red', label ='NON PCA')
PCA_spec_bin_vals, PCA_spec_bins, _ = ax1.hist(PCA_spec_stat, bins='auto', alpha = 0.5, color='blue', label ='PCA')

cv_spec_bin_centers = (cv_spec_bins[:-1] + cv_spec_bins[1:])/2
i_max = np.argmax(cv_spec_bin_vals)
par = [cv_spec_bin_vals[i_max], cv_spec_bin_centers[i_max], 0.05]
popt_cv_spec, pcov_cv_spec = curve_fit(gaussian, cv_spec_bin_centers, cv_spec_bin_vals, par, maxfev=10000)

PCA_spec_bin_centers = (PCA_spec_bins[:-1] + PCA_spec_bins[1:])/2
i_max = np.argmax(PCA_spec_bin_vals)
par = [PCA_spec_bin_vals[i_max], PCA_spec_bin_centers[i_max], 0.05]
popt_PCA_cv_spec, pcov_PCA_cv_spec = curve_fit(gaussian, PCA_spec_bin_centers, PCA_spec_bin_vals, par, maxfev=10000)

x = np.linspace(np.min(np.concatenate((cv_spec_bins, PCA_spec_bins))), np.max(np.concatenate((cv_spec_bins, PCA_spec_bins))), 1000)
ax1.plot(x, gaussian(x,*popt_cv_spec), 'r--', label=f'Gaussian Fit: $\mu$ = {popt_cv_spec[1]:.2f}, $\sigma$ = {popt_cv_spec[2]:.2f}, A = {popt_cv_spec[0]:.2f}')
ax1.plot(x, gaussian(x,*popt_PCA_cv_spec), 'b--', label=f'Gaussian Fit: $\mu$ = {popt_PCA_cv_spec[1]:.2f}, $\sigma$ = {popt_PCA_cv_spec[2]:.2f}, A = {popt_PCA_cv_spec[0]:.2f}')

cross_spec_res = ttest_ind(gaussian(x,*popt_cv_spec), gaussian(x,*popt_PCA_cv_spec), equal_var=False, alternative='greater')
ax1.plot([],[], marker= None, linestyle='None', label=f'p-value fit: {cross_spec_res.pvalue:.2f}')

plt.legend()
plt.title('Specificity Distributions Cross Validation')
plt.savefig('specificity_dist_cross_validation.png')

print('Cross Validation Statistics:')
cross_acc_res = ttest_ind(cross_acc_stat, PCA_acc_stat, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(cross_acc_res.statistic, cross_acc_res.pvalue))
cross_sens_res = ttest_ind(cross_sens_stat, PCA_sens_stat, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value histo: {:.2}'.format(cross_sens_res.statistic, cross_sens_res.pvalue))
cross_spec_res = ttest_ind(cross_spec_stat, PCA_spec_stat, equal_var=False)
print('Specificity t-stat: {:.2}, p-value histo: {:.2}\n'.format(cross_spec_res.statistic, cross_spec_res.pvalue))

fig2, ax2 = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
cv_sens_ext_bin_vals, cv_sens_ext_bins, _ = ax2.hist(cross_sens_ext_stat, bins='auto', alpha = 0.5, color='red', label ='NOT PCA')
PCA_sens_ext_bin_vals, PCA_sens_ext_bins, _ = ax2.hist(PCA_sens_ext_stat, bins='auto', alpha = 0.5, color='blue', label ='PCA')

mask_sens_ext = np.where(cv_sens_ext_bin_vals != 0)
cv_sens_ext_bin_centers = (cv_sens_ext_bins[:-1] + cv_sens_ext_bins[1:])/2
i_max = np.argmax(cv_sens_ext_bin_vals)
par = [cv_sens_ext_bin_vals[i_max], cv_sens_ext_bin_centers[i_max], 0.05]
popt_cv_sens_ext, pcov_cv_sens_ext = curve_fit(gaussian, cv_sens_ext_bin_centers[mask_sens_ext], cv_sens_ext_bin_vals[mask_sens_ext], par, maxfev=10000)

PCA_mask_sens_ext = np.where(PCA_sens_ext_bin_vals != 0)
PCA_sens_ext_bin_centers = (PCA_sens_ext_bins[:-1] + PCA_sens_ext_bins[1:])/2
i_max = np.argmax(PCA_sens_ext_bin_vals)
par = [PCA_sens_ext_bin_vals[i_max], PCA_sens_ext_bin_centers[i_max], 0.05]
popt_PCA_cv_sens_ext, pcov_PCA_cv_sens_ext = curve_fit(gaussian, PCA_sens_ext_bin_centers[PCA_mask_sens_ext], PCA_sens_ext_bin_vals[PCA_mask_sens_ext], par, maxfev=10000)

x = np.linspace(np.min(np.concatenate((cv_sens_ext_bins, PCA_sens_ext_bins))), np.max(np.concatenate((cv_sens_ext_bins, PCA_sens_ext_bins))), 1000)
ax2.plot(x, gaussian(x,*popt_cv_sens_ext), 'r--', label=f'Gaussian Fit: $\mu$ = {popt_cv_sens_ext[1]:.2f}, $\sigma$ = {popt_cv_sens_ext[2]:.2f}, A = {popt_cv_sens_ext[0]:.2f}')
ax2.plot(x, gaussian(x,*popt_PCA_cv_sens_ext), 'b--', label=f'Gaussian Fit: $\mu$ = {popt_PCA_cv_sens_ext[1]:.2f}, $\sigma$ = {popt_PCA_cv_sens_ext[2]:.2f}, A = {popt_PCA_cv_sens_ext[0]:.2f}')

ext_sens_res = ttest_ind(gaussian(x,*popt_cv_sens_ext), gaussian(x,*popt_PCA_cv_sens_ext), equal_var=False, alternative='greater')
ax2.plot([],[], marker= None, linestyle='None', label=f'p-value fit: {ext_sens_res.pvalue:.2f}')

plt.legend()
plt.title('Sensitivity Distributions External Test')
plt.savefig('sensitivity_dist_external_test.png')

fig3, ax3 = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
cv_spec_ext_bin_vals, cv_spec_ext_bins, _ = ax3.hist(cross_spec_ext_stat, bins='auto', alpha = 0.5, color='red', label ='NOT PCA')
PCA_spec_ext_bin_vals, PCA_spec_ext_bins, _ = ax3.hist(PCA_spec_ext_stat, bins='auto', alpha = 0.5, color='blue', label ='PCA')

mask_spec_ext = np.where(cv_spec_ext_bin_vals != 0)
cv_spec_ext_bin_centers = (cv_spec_ext_bins[:-1] + cv_spec_ext_bins[1:])/2
i_max = np.argmax(cv_spec_ext_bin_vals)
par = [cv_spec_ext_bin_vals[i_max], cv_spec_ext_bin_centers[i_max], 0.05]
popt_cv_spec_ext, pcov_cv_spec_ext = curve_fit(gaussian, cv_spec_ext_bin_centers[mask_spec_ext], cv_spec_ext_bin_vals[mask_spec_ext], par, maxfev=10000)

PCA_mask_spec_ext = np.where(PCA_spec_ext_bin_vals != 0)
PCA_spec_ext_bin_centers = (PCA_spec_ext_bins[:-1] + PCA_spec_ext_bins[1:])/2
i_max = np.argmax(PCA_spec_ext_bin_vals)
par = [PCA_spec_ext_bin_vals[i_max], PCA_spec_ext_bin_centers[i_max], 0.05]
popt_PCA_cv_spec_ext, pcov_PCA_cv_spec_ext = curve_fit(gaussian, PCA_spec_ext_bin_centers[PCA_mask_spec_ext], PCA_spec_ext_bin_vals[PCA_mask_spec_ext], par, maxfev=10000)

x = np.linspace(np.min(np.concatenate((cv_spec_ext_bins, PCA_spec_ext_bins))), np.max(np.concatenate((cv_spec_ext_bins, PCA_spec_ext_bins))), 1000)
ax3.plot(x, gaussian(x,*popt_cv_spec_ext), 'r--', label=f'Gaussian Fit: $\mu$ = {popt_cv_spec_ext[1]:.2f}, $\sigma$ = {popt_cv_spec_ext[2]:.2f}, A = {popt_cv_spec_ext[0]:.2f}')
ax3.plot(x, gaussian(x,*popt_PCA_cv_spec_ext), 'b--', label=f'Gaussian Fit: $\mu$ = {popt_PCA_cv_spec_ext[1]:.2f}, $\sigma$ = {popt_PCA_cv_spec_ext[2]:.2f}, A = {popt_PCA_cv_spec_ext[0]:.2f}')

ext_spec_res = ttest_ind(gaussian(x,*popt_cv_spec_ext), gaussian(x,*popt_PCA_cv_spec_ext), equal_var=False, alternative='greater')
ax3.plot([],[], marker= None, linestyle='None', label=f'p-value fit: {ext_spec_res.pvalue:.2f}')

plt.legend()
plt.title('Specificity Distributions External Test')
plt.savefig('specificity_dist_external_test.png')

print('External Test (Majority vote) Statistics:')
ext_acc_res = ttest_ind(cross_acc_ext_stat, PCA_acc_ext_stat, equal_var=False)
print('Accuracy t-stat: {:.2}, p-value: {:.2}'.format(ext_acc_res.statistic, ext_acc_res.pvalue))
ext_sens_res = ttest_ind(cross_sens_ext_stat, PCA_sens_ext_stat, equal_var=False)
print('Sensitivity t-stat: {:.2}, p-value histo: {:.2}'.format(ext_sens_res.statistic, ext_sens_res.pvalue))
ext_spec_res = ttest_ind(cross_spec_ext_stat, PCA_spec_ext_stat, equal_var=False)
print('Specificity t-stat: {:.2}, p-value histo: {:.2}\n'.format(ext_spec_res.statistic, ext_spec_res.pvalue))

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

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")
plt.show()