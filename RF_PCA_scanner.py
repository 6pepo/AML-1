import numpy as np
import sklearn.ensemble as ens
import matplotlib.pyplot as plt
import time as clock
import pandas as pd
from matplotlib import cm, colors
from matplotlib.widgets import Slider
from scipy.io import loadmat
from sklearn.model_selection import KFold

def print_text(ax,x,y,text_list):
    rows,cols = text_list.shape
    for row in range(rows):
        for col in range(cols):
             ax.text(x[col], y[row], f"{text_list[row,col]:.2f}", ha='center', va='center', color='white') 

start = clock.time()

path_file = "01.SVM - RF - Matlab [signal]/data/signal__b.mat"

pc_mat = pd.read_csv("eigenvectors.csv", sep=',', index_col=0)
pc_mat = pc_mat.to_numpy()

data = loadmat(path_file)

g0 = data['g__0'].dot(pc_mat)
g1 = data['g__1'].dot(pc_mat)
tot_pattern = np.concatenate((g0, g1), axis=0)
# int_test_pattern = np.concatenate((g0[18:], g1[18:]), axis=0)

Nneg = len(g0[:, 0])
Npos = len(g1[:, 0])

label0 = 'NEGATIVI'
label1 = 'POSITIVI'
g0_labels = np.full(Nneg, label0)
g1_labels = np.full(Npos, label1)
tot_labels = np.concatenate((g0_labels, g1_labels), axis=0)
# int_test_labels = np.concatenate((g0_labels[18:], g1_labels[18:]), axis=0)

# tree_range =  np.concatenate((range(5, 50, 5), range(50, 100, 20), range(100, 500, 30)), axis = 0)
tree_range = range(10, 410, 10)
tree_range = np.asarray(tree_range)

k_range = range(2, 11, 1)

len_tree = len(tree_range)
len_k = len(k_range)
len_seed = 3

accuracy_list = np.empty((len_tree, len_k, len_seed))
sensitivity_list = np.empty((len_tree, len_k, len_seed))
specificity_list = np.empty((len_tree, len_k, len_seed))

print("Begin Scanning...")

tot_iter = len_tree*len_k*len_seed
iter = 0

for i_seed in range(0,len_seed):
    for i_trees, n_trees in enumerate(tree_range):
        for i_k, k in enumerate(k_range):
            iter += 1
            print("N_trees: ", n_trees, "\tN_fold: ", k, "\tIteration: ", iter, "/", tot_iter,end='\r')     #Python 3.x
            # print("N_trees: {}\tN_fold: {}\tIteration: {}/{} \r".format(n_trees, k, iter, tot_iter)),       #Python 2.x

            # Metrics
            fold_accuracy = []
            fold_sensitivity = []
            fold_specificity = []

            
            kf = KFold(n_splits = k, shuffle = True)
            indices = kf.split(tot_labels)


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

            accuracy = np.mean(fold_accuracy)
            sensitivity = np.mean(fold_sensitivity)
            specificity = np.mean(fold_specificity)

            accuracy_list[i_trees, i_k, i_seed] = accuracy
            sensitivity_list[i_trees, i_k, i_seed] = sensitivity
            specificity_list[i_trees, i_k, i_seed] = specificity

print("\nFinished Scanning!\n")

accuracy_std_list = np.std(accuracy_list, axis=2)
accuracy_list = np.mean(accuracy_list, axis=2)
sensitivity_std_list = np.std(sensitivity_list, axis=2)
sensitivity_list = np.mean(sensitivity_list, axis=2)
specificity_std_list = np.std(specificity_list, axis=2)
specificity_list = np.mean(specificity_list, axis=2)

# fig, ax = plt.subplots(3, 1, sharex=True)
fig, ax = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(0.2, 0.2)

# ax[0].scatter(tree_range, accuracy_list[:, 0], color='C0')
# ax[0].errorbar(tree_range, accuracy_list[:, 0], yerr=accuracy_std_list[:, 0], color='C0')
# ax[0].grid(True)
# ax[0].set_ylim(0, 1.2)
# ax[0].set_title('Accuracy')

# ax[1].scatter(tree_range, sensitivity_list[:, 0], color='C1')
# ax[1].errorbar(tree_range, sensitivity_list[:, 0], yerr=sensitivity_std_list[:, 0], color='C1')
# ax[1].grid(True)
# ax[1].set_ylim(0, 1.2)
# ax[1].set_title('Sensitivity')

# ax[2].scatter(tree_range, specificity_list[:, 0], color='C2')
# ax[2].errorbar(tree_range, specificity_list[:, 0], yerr=specificity_std_list[:, 0], color='C2')
# ax[2].grid(True)
# ax[2].set_ylim(0, 1.2)
# ax[2].set_title('Specificity')

# ax[2].set_xlabel('Number of Trees')

ax[0].scatter(tree_range, sensitivity_list[:, 0], color='C1')
ax[0].errorbar(tree_range, sensitivity_list[:, 0], yerr=sensitivity_std_list[:, 0], color='C1')
ax[0].grid(True)
ax[0].set_ylim(0, 1.2)
ax[0].set_title('Sensitivity')

ax[1].scatter(tree_range, specificity_list[:, 0], color='C2')
ax[1].errorbar(tree_range, specificity_list[:, 0], yerr=specificity_std_list[:, 0], color='C2')
ax[1].grid(True)
ax[1].set_ylim(0, 1.2)
ax[1].set_title('Specificity')

ax[1].set_xlabel('Number of Trees')

axSlide = fig.add_axes([0.075, 0.2, 0.05, 0.7])
kSlide = Slider(ax = axSlide, label = "Number of folds", valmin=k_range[0], valmax=k_range[-1], valstep = 1, valinit=k_range[0], orientation='vertical')

def update(val):
    i = k_range.index(kSlide.val)
    # ax[0].clear()
    # ax[0].scatter(tree_range, accuracy_list[:, i], color='C0')
    # ax[0].errorbar(tree_range, accuracy_list[:, i], yerr=accuracy_std_list[:, i], color='C0')
    # ax[0].grid(True)
    # ax[0].set_ylim(0, 1.2)
    # ax[0].set_title('Accuracy')

    # ax[1].clear()
    # ax[1].scatter(tree_range, sensitivity_list[:, i], color='C1')
    # ax[1].errorbar(tree_range, sensitivity_list[:, i], yerr=sensitivity_std_list[:, i], color='C1')
    # ax[1].grid(True)
    # ax[1].set_ylim(0, 1.2)
    # ax[1].set_title('Sensitivity')

    # ax[2].clear()
    # ax[2].scatter(tree_range, specificity_list[:, i], color='C2')
    # ax[2].errorbar(tree_range, specificity_list[:, i], yerr=specificity_std_list[:, i], color='C2')
    # ax[2].grid(True)
    # ax[2].set_ylim(0, 1.2)
    # ax[2].set_title('Specificity')
    # ax[2].set_xlabel('Number of Trees')

    ax[0].clear()
    ax[0].scatter(tree_range, sensitivity_list[:, i], color='C1')
    ax[0].errorbar(tree_range, sensitivity_list[:, i], yerr=sensitivity_std_list[:, i], color='C1')
    ax[0].grid(True)
    ax[0].set_ylim(0, 1.2)
    ax[0].set_title('Sensitivity')

    ax[1].clear()
    ax[1].scatter(tree_range, specificity_list[:, i], color='C2')
    ax[1].errorbar(tree_range, specificity_list[:, i], yerr=specificity_std_list[:, i], color='C2')
    ax[1].grid(True)
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title('Specificity')
    ax[1].set_xlabel('Number of Trees')

    fig.canvas.draw_idle()

kSlide.on_changed(update)

k_range = np.asarray(k_range)
fig3d = plt.figure()

X, Y = np.meshgrid(k_range, tree_range)

scores = np.concatenate((accuracy_list, sensitivity_list, specificity_list), axis = 0)
normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

# ax_acc = fig3d.add_subplot(131, projection='3d')
# ax_acc.plot_surface(X, Y, accuracy_list, cmap=cm.plasma)
# ax_acc = fig3d.add_subplot(131)
# ax_acc.pcolormesh(k_range, tree_range, accuracy_list, norm=normalization)
# ax_acc.set_ylabel("Number of trees")
# ax_acc.set_xlabel("Number of folds")
# ax_acc.set_title("Accuracy")

# ax_sens = fig3d.add_subplot(132, projection='3d')
# ax_sens.plot_surface(X, Y, sensitivity_list, cmap=cm.plasma)
# ax_sens = fig3d.add_subplot(132)
ax_sens = fig3d.add_subplot(121)
ax_sens.pcolormesh(k_range, tree_range, sensitivity_list, norm=normalization)
print_text(ax_sens, k_range, tree_range, sensitivity_list)
ax_sens.set_ylabel("Number of trees")
ax_sens.set_xlabel("Number of folds")
ax_sens.set_title("Sensitivity")

# ax_spec = fig3d.add_subplot(133, projection='3d')
# ax_spec.plot_surface(X, Y, specificity_list, cmap=cm.plasma)
# ax_spec = fig3d.add_subplot(133)
ax_spec = fig3d.add_subplot(122)
colormesh = ax_spec.pcolormesh(k_range, tree_range, specificity_list, norm=normalization)
print_text(ax_spec, k_range, tree_range, specificity_list)
ax_spec.set_ylabel("Number of trees")
ax_spec.set_xlabel("Number of folds")
ax_spec.set_title("Specificity")

fig3d.colorbar(colormesh,  orientation='vertical')

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")
plt.show()