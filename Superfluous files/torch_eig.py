import numpy as np
import torch 
from scipy.linalg import eig
import time as clock
import sys 

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

dim = 15000
# mat = np.random.rand(dim,dim)
mat = np.array(([1,0,0],[0,1,0],[0,0,1]))

gb = sys.getsizeof(mat)/1024**3

print(f'Start: {clock.localtime().tm_hour}:{clock.localtime().tm_min}:{clock.localtime().tm_sec}')
print(f'Matrix size: {gb} GB')

start = clock.time()
print('SCIPY')
e_val, e_vec = eig(mat)
print(e_val,'\n',e_vec)
print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''\n")

start = clock.time()
print('TORCH')
e_val, e_vec = torch_eig(mat)
print(e_val,'\n',e_vec)
print("Tempo:" + str(round((clock.time() - start)/60)) + "'" + str(round((clock.time() - start)%60)) + "''\n")