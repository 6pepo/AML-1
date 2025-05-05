import numpy as np
import matplotlib.pyplot as plt
import time as clock
import pandas as pd

from matplotlib import cm, colors
from matplotlib.widgets import Slider
from scipy.io import loadmat

import RF_Library as RF

start = clock.time()

pc_mat = pd.read_csv("eigenvectors.csv", sep=',', index_col=0)
pc_mat = pc_mat.to_numpy()

path_file = "signal__b.mat"

data = loadmat(path_file)

g0 = data['g__0'].dot(pc_mat)
g1 = data['g__1'].dot(pc_mat)
tot_pattern = np.concatenate((g0, g1), axis=0)

Nneg = len(g0[:, 0])
Npos = len(g1[:, 0])

label0 = 'NEGATIVI'
label1 = 'POSITIVI'
g0_labels = np.full(Nneg, label0)
g1_labels = np.full(Npos, label1)
tot_labels = np.concatenate((g0_labels, g1_labels), axis=0)

tree_range = range(10, 410, 10)

k_range = range(2, 11, 1)

n_seeds = 100

res = RF.RF_binary_scanner(tree_range, k_range, n_seeds, tot_pattern, tot_labels, label0, label1)

# Single n_fold plot with slider
fig, ax = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(0.2, 0.2)

ax[0].scatter(tree_range, res['Acc List'][:, 0], color='C0')
ax[0].errorbar(tree_range, res['Acc List'][:, 0], yerr=res['Acc Std List'][:, 0], color='C0')
ax[0].grid(True)
ax[0].set_ylim(0, 1)
ax[0].set_title('Accuracy')

ax[1].scatter(tree_range, res['Sens List'][:, 0], color='C1')
ax[1].errorbar(tree_range, res['Sens List'][:, 0], yerr=res['Sens Std List'][:, 0], color='C1')
ax[1].grid(True)
ax[1].set_ylim(0, 1)
ax[1].set_title('Sensitivity')

ax[2].scatter(tree_range, res['Spec List'][:, 0], color='C2')
ax[2].errorbar(tree_range, res['Spec List'][:, 0], yerr=res['Spec Std List'][:, 0], color='C2')
ax[2].grid(True)
ax[2].set_ylim(0, 1)
ax[2].set_title('Specificity')

ax[2].set_xlabel('Number of Trees')

axSlide = fig.add_axes([0.075, 0.2, 0.05, 0.7])
kSlide = Slider(ax = axSlide, label = "Number of folds", valmin=k_range[0], valmax=k_range[-1], valstep = 1, valinit=k_range[0], orientation='vertical')

def update(val):
    i = k_range.index(kSlide.val)
    ax[0].clear()
    ax[0].scatter(tree_range, res['Acc List'][:, i], color='C0')
    ax[0].errorbar(tree_range, res['Acc List'][:, i], yerr=res['Acc Std List'][:, i], color='C0')
    ax[0].grid(True)
    ax[0].set_ylim(0, 1)
    ax[0].set_title('Accuracy')

    ax[1].clear()
    ax[1].scatter(tree_range, res['Sens List'][:, i], color='C1')
    ax[1].errorbar(tree_range, res['Sens List'][:, i], yerr=res['Sens Std List'][:, i], color='C1')
    ax[1].grid(True)
    ax[1].set_ylim(0, 1)
    ax[1].set_title('Sensitivity')

    ax[2].clear()
    ax[2].scatter(tree_range, res['Spec List'][:, i], color='C2')
    ax[2].errorbar(tree_range, res['Spec List'][:, i], yerr=res['Spec Std List'][:, i], color='C2')
    ax[2].grid(True)
    ax[2].set_ylim(0, 1)
    ax[2].set_title('Specificity')
    ax[2].set_xlabel('Number of Trees')

    fig.canvas.draw_idle()

kSlide.on_changed(update)

# Heatmaps n_fold - n_trees
fig3d = plt.figure()

X, Y = np.meshgrid(k_range, tree_range)

scores = np.concatenate((res['Acc List'], res['Sens List'], res['Spec List']), axis = 0)
normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

ax_acc = fig3d.add_subplot(131)
RF.heatmap_plotter(ax_acc, k_range, tree_range, res['Acc List'], "Accuracy", normalization, cm.viridis)

ax_sens = fig3d.add_subplot(132)
# ax_sens = fig3d.add_subplot(121)
RF.heatmap_plotter(ax_sens, k_range, tree_range, res['Sens List'], "Sensitivity", normalization, cm.viridis)

ax_spec = fig3d.add_subplot(133)
# ax_spec = fig3d.add_subplot(122)
colormesh = RF.heatmap_plotter(ax_spec, k_range, tree_range, res['Spec List'], "Specificity", normalization, cm.viridis)


fig3d.colorbar(colormesh,  orientation='vertical')

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")
plt.show()