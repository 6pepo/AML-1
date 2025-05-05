import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import matplotlib.table as tab
import tkinter as tk
import torch

from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

def RF_binary_kfold(n_trees, k, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    
    models = []

    # Metrics of single folds
    fold_accuracy = []
    fold_sensitivity = []
    fold_specificity = []

    kf = KFold(n_splits = k, shuffle = True)
    indices = kf.split(labels)

    if np.any(ext_lab != None):
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
        models.append(model)

        # Internal test
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

        # External test: Majority vote counting
        if np.any(ext_lab != None):
            ext_pred = model.predict(ext_patt)

            for j, pred in enumerate(ext_pred):
                if pred == label0:
                    vote_0_ext[j] += 1
                if pred == label1:
                    vote_1_ext[j] += 1

    # Performance of External Test: Best in Training
    # Best Accuracy
    bacc_model = models[np.argmax(fold_accuracy)]
    bacc_accuracy = 0.
    bacc_sensitivity = 0.
    bacc_specificity = 0.

    if np.any(ext_lab != None):
        bacc_predict = bacc_model.predict(ext_patt)

        for j, pred in enumerate(bacc_predict):
            if pred == label1 and ext_lab[j] == label1:
                bacc_accuracy += 1./(N1_ext+N0_ext)
                bacc_specificity += 1./N0_ext
        
            if pred == label0 and ext_lab[j] == label0:
                bacc_accuracy += 1./(N1_ext+N0_ext)
                bacc_sensitivity += 1./N1_ext

    # Best Sensitivity
    bsens_model = models[np.argmax(fold_sensitivity)]
    bsens_accuracy = 0.
    bsens_sensitivity = 0.
    bsens_specificity = 0.

    if np.any(ext_lab != None):
        bsens_predict = bsens_model.predict(ext_patt)

        for j, pred in enumerate(bsens_predict):
            if pred == label1 and ext_lab[j] == label1:
                bsens_accuracy += 1./(N1_ext+N0_ext)
                bsens_specificity += 1./N0_ext
        
            if pred == label0 and ext_lab[j] == label0:
                bsens_accuracy += 1./(N1_ext+N0_ext)
                bsens_sensitivity += 1./N1_ext

    # Best Specificity
    bspec_model = models[np.argmax(fold_specificity)]
    bspec_accuracy = 0.
    bspec_sensitivity = 0.
    bspec_specificity = 0.
    
    if np.any(ext_lab != None):
        bspec_predict = bspec_model.predict(ext_patt)

        for j, pred in enumerate(bspec_predict):
            if pred == label1 and ext_lab[j] == label1:
                bspec_accuracy += 1./(N1_ext+N0_ext)
                bspec_specificity += 1./N0_ext
        
            if pred == label0 and ext_lab[j] == label0:
                bspec_accuracy += 1./(N1_ext+N0_ext)
                bspec_sensitivity += 1./N1_ext

    # External test: Majority vote
    ext_accuracy = 0.
    ext_sensitivity = 0.
    ext_specificity = 0.
    conf_mat = np.zeros((2,2))

    if np.any(ext_lab != None):
        for i, true_label in enumerate(ext_lab):
            if vote_0_ext[i]>=vote_1_ext[i]:
                if true_label == label1:
                    conf_mat[1][0] += 1
                if true_label == label0:
                    ext_accuracy += 1./(N1_ext+N0_ext)
                    ext_specificity += 1./N0_ext
                    conf_mat[1][1] += 1
            if vote_0_ext[i]<vote_1_ext[i]:
                if true_label == label1:
                    ext_accuracy += 1./(N1_ext+N0_ext)
                    ext_sensitivity += 1./N1_ext
                    conf_mat[0][0] += 1
                if true_label == label0:
                    conf_mat[0][1] += 1

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
        'Ext Spec': ext_specificity,
        'BAcc Acc': bacc_accuracy,
        'BAcc Sens': bacc_sensitivity,
        'BAcc Spec': bacc_specificity,
        'BSens Acc': bsens_accuracy,
        'BSens Sens': bsens_sensitivity,
        'BSens Spec': bsens_specificity,
        'BSpec Acc': bspec_accuracy,
        'BSpec Sens': bspec_sensitivity,
        'BSpec Spec': bspec_specificity,
        'Models': models,
        'Conf Mat': conf_mat
    }

    return res

def RF_binary_scanner(tree_range, k_range, n_seeds, patterns, labels, label0, label1, ext_patt = None, ext_lab = None):
    len_tree = len(tree_range)
    len_k = len(k_range)

    accuracy_list = np.empty((len_tree, len_k, n_seeds))
    sensitivity_list = np.empty((len_tree, len_k, n_seeds))
    specificity_list = np.empty((len_tree, len_k, n_seeds))

    print("Begin Scanning...")

    tot_iter = len_tree*len_k*n_seeds
    progress = tqdm(total=tot_iter)
    iter = 0

    for i in range(n_seeds):
        for i_trees, n_trees in enumerate(tree_range):
            for i_k, k in enumerate(k_range):
                iter += 1
                
                # Progress bar
                progress.set_description(f"Trees: {i_trees+1}/{len_tree} | Fold: {i_k+1}/{len_k} | Rand_state: {i+1}/{n_seeds}")
                progress.update(1)
                
                # print("N_trees: ", n_trees, "\tN_fold: ", k, "\tIteration: ", iter, "/", tot_iter,end='\r')     #Python 3.x
                # print("N_trees: {}\tN_fold: {}\tIteration: {}/{} \r".format(n_trees, k, iter, tot_iter)),       #Python 2.x

                res = RF_binary_kfold(n_trees, k, patterns, labels, label0, label1)

                accuracy_list[i_trees, i_k, i] = res['Acc']
                sensitivity_list[i_trees, i_k, i] = res['Sens']
                specificity_list[i_trees, i_k, i] = res['Spec']

    print("\nFinished Scanning!                                                \n")

    res = {
        'Acc List': np.mean(accuracy_list, axis=2),
        'Acc Std List': np.std(accuracy_list, axis=2),
        'Sens List': np.mean(sensitivity_list, axis=2),
        'Sens Std List': np.std(sensitivity_list, axis=2),
        'Spec List': np.mean(specificity_list, axis=2),
        'Spec Std List': np.std(specificity_list, axis=2)
    }

    return res

def confMat_binary_plot(conf_mat, accuracy=None, sensitivity=None, specificity=None, precision=None, title=None):
    fig, ax = plt.subplots()
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

    norm = colors.Normalize(vmin = 0, vmax = np.max(conf_mat))
    normalized = norm((0., conf_mat[0][0], conf_mat[0][1], conf_mat[1][0], conf_mat[1][1]))
    cell_color = cm.viridis(normalized)
    middle = np.max(conf_mat)/2
    col_max = np.argmax(conf_mat) + 1
    text_color = np.empty((2,2,4))
    for i in (0,1):
        for j in (0,1):
            if conf_mat[i][j] < middle:
                text_color[i][j] = cell_color[col_max]
            else:
                text_color[i][j] = cell_color[0]
                
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
    table.add_cell(1,2, width = 0.3, height = 0.2, text='Positive\n\n'+str(conf_mat[0][0]+conf_mat[1][0]), loc='center', fontproperties = fontproperties)
    table.add_cell(1,3, width = 0.3, height = 0.2, text='Negative\n\n'+str(conf_mat[0][1]+conf_mat[1][1]), loc='center', fontproperties = fontproperties)

    table.add_cell(2,0, width = 0.1, height = 0.3, text='Classifier', loc='center', fontproperties = fontproperties)
    table[2,0].set_text_props(rotation = 'vertical')
    table[2,0].visible_edges = 'LTR'
    table.add_cell(2,1, width = 0.2, height = 0.3, text='Positive\n\n'+str(round(conf_mat[0][0]+conf_mat[0][1], 2)), loc='center', fontproperties = fontproperties)
    table.add_cell(2,2, width = 0.3, height = 0.3, text=str(conf_mat[0][0]), loc='center', fontproperties = fontproperties, facecolor = cell_color[1])
    table[2,2].set_text_props(c = text_color[0][0])
    table.add_cell(2,3, width = 0.3, height = 0.3, text=str(conf_mat[0][1]), loc='center', fontproperties = fontproperties, facecolor = cell_color[2])
    table[2,3].set_text_props(c = text_color[0][1])
    table.add_cell(2,4, width = 0.2, height = 0.3, text='Precision\n\n{:.2}'.format(precision), loc='center', fontproperties = fontproperties)

    table.add_cell(3,0, width = 0.1, height = 0.3, text='Output of', loc='center', fontproperties = fontproperties)
    table[3,0].set_text_props(rotation = 'vertical')
    table[3,0].visible_edges = 'BLR'
    table.add_cell(3,1, width = 0.2, height = 0.3, text='Negative\n\n'+str(round(conf_mat[1][0]+conf_mat[1][1], 2)), loc='center', fontproperties = fontproperties)
    table.add_cell(3,2, width = 0.3, height = 0.3, text=str(conf_mat[1][0]), loc='center', fontproperties = fontproperties, facecolor = cell_color[3])
    table[3,2].set_text_props(c = text_color[1][0])
    table.add_cell(3,3, width = 0.3, height = 0.3, text=str(conf_mat[1][1]), loc='center', fontproperties = fontproperties, facecolor = cell_color[4])
    table[3,3].set_text_props(c = text_color[1][1])
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
    # root = tk.Tk()
    # screen_width = root.winfo_screenwidth()
    # screen_height = root.winfo_screenheight()
    # root.destroy() 

    # fig, ax = plt.subplots(figsize=(screen_width / 100, screen_height / 100))
    fig, ax = plt.subplots()
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

    ax.legend(loc='best')

    return fig, ax, stat_res

def torch_eig(mat):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Torch Device: ', device.type)

    torch_mat = torch.from_numpy(mat).to(device, dtype=torch.float32)

    e_val, e_vec = torch.linalg.eigh(torch_mat)

    e_val = e_val.cpu().numpy()
    e_vec = e_vec.cpu().numpy()

    torch.cuda.empty_cache()

    return e_val, e_vec

def heatmap_plotter(ax, x, y, array, title, norm, cmap = cm.viridis):
    colormesh = ax.pcolormesh(x, y, array, norm=norm, cmap=cmap)
    ax.set_ylabel("Number of trees")
    ax.set_xlabel("Number of folds")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    rows,cols = array.shape
    for row in range(rows):
        for col in range(cols):
            if norm(array[row][col]) < 0.5:
                ax.text(x[col], y[row], f"{array[row,col]:.2f}", ha='center', va='center', color=cmap(0.99))
            else:
                ax.text(x[col], y[row], f"{array[row,col]:.2f}", ha='center', va='center', color=cmap(0.))

    return colormesh
