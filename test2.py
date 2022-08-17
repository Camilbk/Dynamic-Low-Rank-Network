from data import cifar10, fashionMnist
from matplotlib import pyplot as plt
import torch
from networks import DynTensorResNet, ProjTensorResNet
from optimisation import train
import numpy as np

N = 150
V = 150
batch_size = 3

L = 5
max_epochs = 3

plt.rcParams.update({
    "font.size":25})


#### DYNAMIC LOW-RANK NET
# DATA
data = cifar10( N, V, batch_size, k=3, transform='tucker')

# NETWORK
# CONSTRUCT NETWORK
dyn = DynTensorResNet(data, L, use_cayley=False)
proj = ProjTensorResNet(data, L)

# TRAIN NETWORK
#torch.autograd.set_detect_anomaly(True)
_, Dyn_acc_train_mnist, _, Dyn_acc_val_mnist = train(dyn,  max_epochs = max_epochs)
_, Dyn_acc_train_fashion, _, Dyn_acc_val_fashion = train(proj,  max_epochs = max_epochs)


print("DynResNet")
print(max(Dyn_acc_train_mnist))
print(max(Dyn_acc_val_mnist))
print("\n")
print("ProjTensorNet")
print(max(Dyn_acc_train_fashion))
print(max(Dyn_acc_val_fashion))
print("\n")



s, err_U_Proj_mnist, err_V_Proj_mnist = dyn.orthogonality
print(s)
s, err_U_Proj_fashion, err_V_Proj_fashion = proj.orthogonality
print(s)
print("\n")


plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_U_Proj_mnist)), err_U_Proj_mnist, 'tab:green', label = 'dyn')
plt.plot(range(len(err_U_Proj_fashion)), err_U_Proj_fashion, 'tab:cyan', label = 'proj' )
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - U^T U ||_F$')
#plt.savefig('MNISTvsFashionMNIST_orthogonalityU%i.png'  %L, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_V_Proj_mnist)), err_V_Proj_mnist, 'tab:green', label = 'dyn')
plt.plot(range(len(err_V_Proj_fashion)), err_V_Proj_fashion, 'tab:cyan', label = 'proj' )
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - V^T V ||_F$')
#plt.savefig('MNISTvsFashionMNIST_orthogonalityV%i.png' %L,  bbox_inches='tight')
plt.show()

