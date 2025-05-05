import numpy as np
import time as clock
import pandas as pd

import RF_Library as RF

from sklearn.model_selection import KFold
from scipy.io import loadmat

start = clock.time()

path_file = "signal__b.mat"

data = loadmat(path_file)

g0 = data['g__0']
g1 = data['g__1']

print("Calculating correlation matrix...")
tot_pattern = np.concatenate((g0, g1), axis=0)
corr = np.corrcoef(tot_pattern, rowvar=False)

print('Calculating eigenvalues and eigenvectors...')
e_val, e_vec = RF.torch_eig(corr) 

print('Sorting eigenvalues and eigenvectors...\n')
sort_index = np.argsort(e_val)[::-1]        # First 27 eigenvalues and eigenvectors are confirmed to be fully real
e_val = e_val[sort_index].real
e_vec = e_vec[:, sort_index].real

for i in range(len(e_val[:5])):
    print('Eigenvalue: {}'.format(e_val[i]))
    print('Eigenvector: {}\n'.format(e_vec[:5, i]))

e_val_sum = np.sum(e_val)

val_sum = 0
n_vec = 0
perc_tresh = 0.9            # Threshold of features at wich we cut
print('\nOriginal\tPercent\tCumulative')
for i,val in enumerate(e_val):
    val_sum += val
    print('{}\t{:.2%}\t{:.2%}'.format(sort_index[i], val/e_val_sum, val_sum/e_val_sum))
    if val_sum/e_val_sum > perc_tresh:
        n_vec = i+1
        break

print("Searching primary contributors of eigenvalues")
labels = []
pc = []
for i in range(n_vec):
    prim_coeff = np.argsort(abs(e_vec[:, i]))[::-1]
    pc.append(prim_coeff)
    labels.append('v'+str(i+1))

df_pc = pd.DataFrame(data = np.asarray(pc)[:, :n_vec], columns=np.asarray(labels))
df_corr = pd.DataFrame(data = corr[:n_vec, :n_vec])
df_eval = pd.DataFrame(data = e_val[:n_vec])
df_evec = pd.DataFrame(data = e_vec[:, :n_vec])
df_pc.to_csv('primaryCoeff.csv', sep = ',')
df_corr.to_csv('corrMatrix.csv', sep = ',')
df_eval.to_csv('eigenvalues.csv', sep = ',')
df_evec.to_csv('eigenvectors.csv', sep = ',')

print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''")