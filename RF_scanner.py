import numpy as np
import matplotlib.pyplot as plt
import time as clock

from matplotlib import cm, colors
from matplotlib.widgets import Slider
from scipy.io import loadmat

import RF_Library as RF

start = clock.time()

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

tree_range = range(10, 410, 10)
tree_range = np.asarray(tree_range)

k_range = range(2, 11, 1)

res = RF.RF_binary_scanner(tree_range, k_range, tot_pattern, tot_labels, label0, label1)

fig, ax = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(0.2, 0.2)

ax[0].scatter(tree_range, res['Acc List'][:, 0], color='C0')
ax[0].errorbar(tree_range, res['Acc List'][:, 0], yerr=res['Acc Err List'][:, 0], color='C0')
ax[0].grid(True)
ax[0].set_ylim(0, 1.2)
ax[0].set_title('Accuracy')

ax[1].scatter(tree_range, res['Sens List'][:, 0], color='C1')
ax[1].errorbar(tree_range, res['Sens List'][:, 0], yerr=res['Sens Err List'][:, 0], color='C1')
ax[1].grid(True)
ax[1].set_ylim(0, 1.2)
ax[1].set_title('Sensitivity')

ax[2].scatter(tree_range, res['Spec List'][:, 0], color='C2')
ax[2].errorbar(tree_range, res['Spec List'][:, 0], yerr=res['Spec Err List'][:, 0], color='C2')
ax[2].grid(True)
ax[2].set_ylim(0, 1.2)
ax[2].set_title('Specificity')

ax[2].set_xlabel('Number of Trees')

axSlide = fig.add_axes([0.075, 0.2, 0.05, 0.7])
kSlide = Slider(ax = axSlide, label = "Number of folds", valmin=k_range[0], valmax=k_range[-1], valstep = 1, valinit=k_range[0], orientation='vertical')

def update(val):
    i = k_range.index(kSlide.val)
    ax[0].clear()
    ax[0].scatter(tree_range, res['Acc List'][:, i], color='C0')
    ax[0].errorbar(tree_range, res['Acc List'][:, i], yerr=res['Acc Err List'][:, i], color='C0')
    ax[0].grid(True)
    ax[0].set_ylim(0, 1.2)
    ax[0].set_title('Accuracy')

    ax[1].clear()
    ax[1].scatter(tree_range, res['Sens List'][:, i], color='C1')
    ax[1].errorbar(tree_range, res['Sens List'][:, i], yerr=res['Sens Err List'][:, i], color='C1')
    ax[1].grid(True)
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title('Sensitivity')

    ax[2].clear()
    ax[2].scatter(tree_range, res['Spec List'][:, i], color='C2')
    ax[2].errorbar(tree_range, res['Spec List'][:, i], yerr=res['Spec Err List'][:, i], color='C2')
    ax[2].grid(True)
    ax[2].set_ylim(0, 1.2)
    ax[2].set_title('Specificity')
    ax[2].set_xlabel('Number of Trees')

    fig.canvas.draw_idle()

kSlide.on_changed(update)

fig3d = plt.figure()

X, Y = np.meshgrid(k_range, tree_range)

scores = np.concatenate((res['Acc List'], res['Sens List'], res['Spec List']), axis = 0)
normalization = colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))

ax_acc = fig3d.add_subplot(131)
ax_acc.pcolormesh(k_range, tree_range, res['Acc List'], norm=normalization, cmap=cm.plasma)
ax_acc.set_ylabel("Number of trees")
ax_acc.set_xlabel("Number of folds")
ax_acc.set_title("Accuracy")

ax_sens = fig3d.add_subplot(132)
ax_sens.pcolormesh(k_range, tree_range, res['Sens List'], norm=normalization,cmap=cm.plasma)
ax_sens.set_ylabel("Number of trees")
ax_sens.set_xlabel("Number of folds")
ax_sens.set_title("Sensitivity")

ax_spec = fig3d.add_subplot(133)
colormesh = ax_spec.pcolormesh(k_range, tree_range, res['Spec List'], norm=normalization, cmap=cm.plasma)
ax_spec.set_ylabel("Number of trees")
ax_spec.set_xlabel("Number of folds")
ax_spec.set_title("Specificity")

fig3d.colorbar(colormesh,  orientation='vertical')
print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")
plt.show()