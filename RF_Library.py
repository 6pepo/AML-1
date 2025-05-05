import numpy as np
import sklearn.ensemble as ens
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.table as tab
import tkinter as tk

from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind

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


def confMat_binary_plot(conf_mat, accuracy=None, sensitivity=None, specificity=None, precision=None, title=None):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    cmap = plt.get_cmap('bwr')
    norm = mcolors.Normalize(vmin=np.min(conf_mat),vmax=np.max(conf_mat))
    fig, ax = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
    if title == None:
        ax.set_title("Confusion matrix")
    else:
        ax.set_title(title)
    ax.axis('off')
    fig.frameon = False
    
    tot = conf_mat[0][0]+conf_mat[0][1]+conf_mat[1][0]+conf_mat[1][1]
    if precision == None:
        precision = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1])
    if sensitivity == None:
        sensitivity = conf_mat[0][0] / (conf_mat[0][0]+conf_mat[1][0])
    if specificity == None:
        specificity = conf_mat[1][1] / (conf_mat[1][1]+conf_mat[0][1])
    if accuracy == None:
        accuracy = (conf_mat[0][0] + conf_mat[1][1]) / tot
    F1score = 2. * (precision * sensitivity) / (precision + sensitivity)

    table = tab.Table(ax, loc='upper center')
    table.auto_set_font_size(False)
    fontproperties = {'family': 'sans-serif',
                      'style': 'normal',
                      'variant': 'normal',
                      'stretch': 'normal',
                      'weight': 'normal',
                      'size': 'medium',
                      'math_fontfamily': 'dejavusans'}

    table.add_cell(0,2, width = 0.3, height = 0.1, text='Actual', loc='right', fontproperties = fontproperties)
    table[0,2].visible_edges = 'BTL'
    table.add_cell(0,3, width = 0.3, height = 0.1, text='Condition', loc='left', fontproperties = fontproperties)
    table[0,3].visible_edges = 'BTR'

    table.add_cell(1,1, width = 0.2, height = 0.2, text='Total\n\n'+str(tot), loc='center', fontproperties = fontproperties)
    table.add_cell(1,2, width = 0.3, height = 0.2, text='Positive\n\n'+str(round(conf_mat[0][0]+conf_mat[1][0],2)), loc='center', fontproperties = fontproperties)
    table.add_cell(1,3, width = 0.3, height = 0.2, text='Negative\n\n'+str(round(conf_mat[0][1]+conf_mat[1][1],2)), loc='center', fontproperties = fontproperties)

    table.add_cell(2,0, width = 0.1, height = 0.3, text='Classifier', loc='center', fontproperties = fontproperties)
    table[2,0].set_text_props(rotation = 'vertical')
    table[2,0].visible_edges = 'LTR'
    table.add_cell(2,1, width = 0.2, height = 0.3, text='Positive\n\n'+str(conf_mat[0][0]+conf_mat[0][1]), loc='center', fontproperties = fontproperties)
    table.add_cell(2,2, width = 0.3, height = 0.3, text=str(round(conf_mat[0][0],2)), loc='center', fontproperties = fontproperties, facecolor='white')
    table.add_cell(2,3, width = 0.3, height = 0.3, text=str(round(conf_mat[0][1],2)), loc='center', fontproperties = fontproperties, facecolor='white')
    table.add_cell(2,4, width = 0.2, height = 0.3, text='Precision\n\n{:.2}'.format(precision), loc='center', fontproperties = fontproperties)

    table.add_cell(3,0, width = 0.1, height = 0.3, text='Output of', loc='center', fontproperties = fontproperties)
    table[3,0].set_text_props(rotation = 'vertical')
    table[3,0].visible_edges = 'BLR'
    table.add_cell(3,1, width = 0.2, height = 0.3, text='Negative\n\n'+str(conf_mat[1][0]+conf_mat[1][1]), loc='center', fontproperties = fontproperties)
    table.add_cell(3,2, width = 0.3, height = 0.3, text=str(conf_mat[1][0]), loc='center', fontproperties = fontproperties, facecolor='white')
    table.add_cell(3,3, width = 0.3, height = 0.3, text=str(conf_mat[1][1]), loc='center', fontproperties = fontproperties, facecolor='white')
    table.add_cell(3,4, width = 0.2, height = 0.3)

    table.add_cell(4,2, width = 0.3, height = 0.2, text='Sensitivity\n\n{:.2}'.format(sensitivity), loc='center', fontproperties = fontproperties)
    table.add_cell(4,3, width = 0.3, height = 0.2, text='Specificity\n\n{:.2}'.format(specificity), loc='center', fontproperties = fontproperties)
    table.add_cell(4,4, width = 0.2, height = 0.2, text='Acc: {:.2}\n\nF1: {:.2}'.format(accuracy, F1score), loc='center', fontproperties = fontproperties)
    
    ax.add_table(table)

    return fig, ax

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

def hp_mode(mu1,mu2):
    if mu1 < mu2:
        return 'less'
    else:
        return 'greater'
    
def check_var(sig1,sig2):
    if sig1==sig2:
        return True
    else:
        return False

def plot_histo_gaus_stat(dist1, label1, dist2, label2):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy() 

    fig, ax = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
    bin_vals1, bins1, _ = ax.hist(dist1, bins='auto', alpha = 0.5, color='red', label = label1)
    bin_vals2, bins2, _ = ax.hist(dist2, bins='auto', alpha = 0.5, color='blue', label = label2)

    mask1 = np.where(bin_vals1 != 0)
    bin_centers1 = (bins1[:-1] + bins1[1:])/2
    i_max1 = np.argmax(bin_vals1)
    par1 = [bin_vals1[i_max1], bin_centers1[i_max1], 0.05]
    popt1, pcov1 = curve_fit(gaussian, bin_centers1[mask1], bin_vals1[mask1], par1, maxfev=10000)

    mask2 = np.where(bin_vals2 != 0)
    bin_centers2 = (bins2[:-1] + bins2[1:])/2
    i_max2 = np.argmax(bin_vals2)
    par2 = [bin_vals2[i_max2], bin_centers2[i_max2], 0.05]
    popt2, pcov2 = curve_fit(gaussian, bin_centers2[mask2], bin_vals2[mask2], par2, maxfev=10000)
    
    x = np.linspace(np.min(np.concatenate((bins1, bins2))), np.max(np.concatenate((bins1,bins2))), 1000)
    ax.plot(x, gaussian(x,*popt1), 'r--', label=f'Gaussian Fit: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}')
    ax.plot(x, gaussian(x,*popt2), 'b--', label=f'Gaussian Fit: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}')

    stat_res = ttest_ind(dist1, dist2 ,equal_var=check_var(popt1[2], popt2[2]), alternative=hp_mode(popt1[1], popt2[1]))


    ax.plot([],[], marker= None, linestyle='None', label=f't-stat: {stat_res.statistic:.2f}, p-value: {stat_res.pvalue:.2f}')

    ax.legend(loc='best', fontsize='large')

    return fig,ax,stat_res
