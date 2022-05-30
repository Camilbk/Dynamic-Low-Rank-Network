from data import mnist, fashionMnist
from matplotlib import pyplot as plt
import torch
from networks import ResNet, ProjResNet, DynResNet
from optimisation import train
import numpy as np

N = 5000
V = 1500
batch_size = 30

L = 100
max_epochs = 20

plt.rcParams.update({
    "font.size":25})


#### DYNAMIC LOW-RANK NET
# DATA
data_mnist_svd = mnist( N, V, batch_size, k=3, transform='svd')
data_fashion_svd = fashionMnist( N, V, batch_size, k=3, transform='svd')
# NETWORK
# CONSTRUCT NETWORK
DynResNet_mnist = DynResNet(data_mnist_svd, L, d_hat='none', use_cayley=False)
DynResNet_fashion = DynResNet(data_fashion_svd, L, d_hat='none', use_cayley=False)
# TRAIN NETWORK
#torch.autograd.set_detect_anomaly(True)
_, Dyn_acc_train_mnist, _, Dyn_acc_val_mnist = train(DynResNet_mnist,  max_epochs = max_epochs)
_, Dyn_acc_train_fashion, _, Dyn_acc_val_fashion = train(DynResNet_fashion,  max_epochs = max_epochs)
#save models
PATH_mnist = "../../Models/mnist_dynnet100.pt"
PATH_fashion = "../../Models/fashion_dynnet100.pt"
torch.save(DynResNet_mnist.state_dict(), PATH_mnist)
torch.save(DynResNet_fashion.state_dict(), PATH_fashion)



#### STANDARD RESNET
# DATA
data_mnist = mnist( N, V, batch_size, k=28, transform='none')
data_fashion = fashionMnist( N, V, batch_size, k=28, transform='none')
# NETWORK
# CONSTRUCT NETWORK
net_mnist = ResNet(data_mnist, L, trainable_stepsize=True, d_hat='none')
net_fashion = ResNet(data_fashion, L, trainable_stepsize=True, d_hat='none')
# TRAIN NETWORK
torch.autograd.set_detect_anomaly(True)
_, acc_train_mnist, _, acc_val_mnist = train(net_mnist,  max_epochs = max_epochs)
_, acc_train_fashion, _, acc_val_fashion = train(net_fashion,  max_epochs = max_epochs)
#save models
PATH_mnist = "../../Models/mnist_resnet100.pt"
PATH_fashion = "../../Models/fashion_resnet100.pt"
torch.save(net_mnist.best_state, PATH_mnist)
torch.save(net_fashion.best_state, PATH_fashion)


#### PROJECTION RESNET
# DATA
#data_mnist_svd = mnist( N, V, batch_size, k=3, transform='svd')
#data_fashion_svd = fashionMnist( N, V, batch_size, k=3, transform='svd')
# NETWORK
# CONSTRUCT NETWORK
ProjNet_mnist = ProjResNet(data_mnist_svd, L, trainable_stepsize=True, d_hat='none')
ProjNet_fashion = ProjResNet(data_fashion_svd, L, trainable_stepsize=True, d_hat='none')
# TRAIN NETWORK
torch.autograd.set_detect_anomaly(True)
_, Proj_acc_train_mnist, _, Proj_acc_val_mnist = train(ProjNet_mnist,  max_epochs = max_epochs)
_, Proj_acc_train_fashion, _, Proj_acc_val_fashion = train(ProjNet_fashion,  max_epochs = max_epochs)

#save models
PATH_mnist = "../../Models/mnist_projnet100.pt"
PATH_fashion = "../../Models/fashion_projnet100.pt"
torch.save(ProjNet_mnist.state_dict(), PATH_mnist)
torch.save(ProjNet_fashion.state_dict(), PATH_fashion)

### Nice

print("\n")
print("ResNet")
print("mnist")
print(max(acc_train_mnist))
print(max(acc_val_mnist))
print("\n")
print("fashion")
print(max(acc_train_fashion))
print(max(acc_val_fashion))
print("\n")
print("ProjResNet")
print("mnist")
print(max(Proj_acc_train_mnist))
print(max(Proj_acc_val_mnist))
print("\n")
print("fashion")
print(max(Proj_acc_train_fashion))
print(max(Proj_acc_val_fashion))
print("\n")
print("DynResNet")
print("mnist")
print(max(Dyn_acc_train_mnist))
print(max(Dyn_acc_val_mnist))
print("\n")
print("fashion")
print(max(Dyn_acc_train_fashion))
print(max(Dyn_acc_val_fashion))
print("\n")


# plot accurracy

plt.figure(figsize=(12, 8), dpi=80)
### ResNet
plt.plot(range(len(acc_train_mnist)), acc_train_mnist, 'tab:orange', label = 'mnist')
plt.plot(range(len(acc_val_mnist)), acc_val_mnist, 'tab:orange', linestyle='--')

plt.plot(range(len(acc_train_fashion)), acc_train_fashion, 'tab:purple', label = 'fashion' )
plt.plot(range(len(acc_val_fashion)), acc_val_fashion, 'tab:purple', linestyle='--')

# ProjNet
plt.plot(range(len(Proj_acc_train_mnist)), Proj_acc_train_mnist, 'tab:green', label = 'Proj. mnist')
plt.plot(range(len(Proj_acc_val_mnist)), Proj_acc_val_mnist, 'tab:green', linestyle='--')

plt.plot(range(len(Proj_acc_train_fashion)), Proj_acc_train_fashion, 'tab:cyan', label = 'Proj. fashion' )
plt.plot(range(len(Proj_acc_val_fashion)), Proj_acc_val_fashion, 'tab:cyan', linestyle='--')

# DynNet
plt.plot(range(len(Dyn_acc_train_mnist)), Dyn_acc_train_mnist, 'tab:brown', label = 'Dyn. mnist' )
plt.plot(range(len(Dyn_acc_val_mnist)), Dyn_acc_val_mnist, 'tab:brown', linestyle='--')

plt.plot(range(len(Dyn_acc_train_fashion)), Dyn_acc_train_fashion, 'tab:gray', label = 'Dyn. fashion' )
plt.plot(range(len(Dyn_acc_val_fashion)), Dyn_acc_val_fashion, 'tab:gray', linestyle='--')

plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'MNIST vs FashionMNIST ')
plt.savefig('MNISTvsFashionMNIST_accuracy100.png', bbox_inches='tight')
plt.show()

# plot orthogonality

s, err_U_Proj_mnist, err_V_Proj_mnist = ProjNet_mnist.orthogonality
print(s)
s, err_U_Proj_fashion, err_V_Proj_fashion = ProjNet_fashion.orthogonality
print(s)
print("\n")

s, err_U_Dyn_mnist, err_V_Dyn_mnist = DynResNet_mnist.orthogonality
print(s)
s, err_U_Dyn_fashion, err_V_Dyn_fashion = DynResNet_fashion.orthogonality
print(s)


plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_U_Proj_mnist)), err_U_Proj_mnist, 'tab:green', label = 'Proj. mnist')
plt.plot(range(len(err_U_Proj_fashion)), err_U_Proj_fashion, 'tab:cyan', label = 'Proj. fashion' )
# DynNet
plt.plot(range(len(err_U_Dyn_mnist)), err_U_Dyn_mnist, 'tab:brown', label = 'Dyn. mnist' )
plt.plot(range(len(err_U_Dyn_fashion)), err_U_Dyn_fashion, 'tab:gray', label = 'Dyn. fashion' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - U^T U ||_F$')
plt.savefig('MNISTvsFashionMNIST_orthogonalityU100.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_V_Proj_mnist)), err_V_Proj_mnist, 'tab:green', label = 'Proj. mnist')
plt.plot(range(len(err_V_Proj_fashion)), err_V_Proj_fashion, 'tab:cyan', label = 'Proj. fashion' )
# DynNet
plt.plot(range(len(err_V_Dyn_mnist)), err_V_Dyn_mnist, 'tab:brown', label = 'Dyn. mnist' )
plt.plot(range(len(err_V_Dyn_fashion)), err_V_Dyn_fashion, 'tab:gray', label = 'Dyn. fashion' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - V^T V ||_F$')
plt.savefig('MNISTvsFashionMNIST_orthogonalityV100.png', bbox_inches='tight')
plt.show()


# Rank evolution

ranks_mnist = net_mnist.rank_evolution
ranks_fashion = net_fashion.rank_evolution

ranks_proj_mnist = ProjNet_mnist.rank_evolution
ranks_proj_fashion = ProjNet_fashion.rank_evolution

ranks_dyn_mnist = DynResNet_mnist.rank_evolution
ranks_dyn_fashion = DynResNet_fashion.rank_evolution


plt.figure(figsize=(12, 8), dpi=80)

### ResNet
plt.plot(range(len(ranks_mnist)), ranks_mnist, 'tab:orange', linewidth=4, label = 'mnist')
plt.plot(range(len(ranks_fashion)), ranks_fashion, 'tab:purple',linewidth=1, label = 'fashion' )
# ProjNet
plt.plot(range(len(ranks_proj_mnist)), ranks_proj_mnist, 'tab:green',linewidth=10, label = 'Proj. mnist')
plt.plot(range(len(ranks_proj_fashion)), ranks_proj_fashion, 'tab:cyan', linewidth=7,  label = 'Proj. fashion' )
# DynNet
plt.plot(range(len(ranks_dyn_mnist)), ranks_dyn_mnist, 'tab:brown', linewidth=4, label = 'Dyn. mnist' )
plt.plot(range(len(ranks_dyn_fashion)), ranks_dyn_fashion, 'tab:gray',linewidth=1, label = 'Dyn. fashion' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r'rank$(X)$')
plt.savefig('MNISTvsFashionMNIST_ranks100.png', bbox_inches='tight')
plt.show()




# integration error ..  ?

Ierr_U_Proj_mnist, Ierr_V_Proj_mnist = ProjNet_mnist.get_integration_error
Ierr_U_Proj_fashion, Ierr_V_Proj_fashion = ProjNet_fashion.get_integration_error

Ierr_U_Dyn_mnist, Ierr_V_Dyn_mnist = DynResNet_mnist.get_integration_error
Ierr_U_Dyn_fashion, Ierr_V_Dyn_fashion = DynResNet_fashion.get_integration_error

print("integration error U Proj MNIST: ", Ierr_U_Proj_mnist)
print("integration error V Proj MNIST: ", Ierr_V_Proj_mnist)

print("integration error U Proj fashion: ", Ierr_U_Proj_fashion)
print("integration error V Proj fashion: ", Ierr_V_Proj_fashion)

print("integration error U Proj MNIST: ", Ierr_U_Dyn_mnist)
print("integration error V Proj MNIST: ", Ierr_V_Dyn_mnist)

print("integration error U Proj fashion: ", Ierr_U_Dyn_fashion)
print("integration error V Proj fashion: ", Ierr_V_Dyn_fashion)

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(Ierr_U_Proj_mnist)), Ierr_U_Proj_mnist, 'tab:green', label = 'Proj. mnist')
plt.plot(range(len(Ierr_U_Proj_fashion)), Ierr_U_Proj_fashion, 'tab:cyan', label = 'Proj. fashion' )
# DynNet
plt.plot(range(len(Ierr_U_Dyn_mnist)), Ierr_U_Dyn_mnist, 'tab:brown', label = 'Dyn. mnist' )
plt.plot(range(len(Ierr_U_Dyn_fashion)), Ierr_U_Dyn_fashion, 'tab:gray', label = 'Dyn. fashion' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| U - \tilde U ||_F$')
plt.savefig('MNISTvsFashionMNIST_integrationerrorU100.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(Ierr_V_Proj_mnist)), Ierr_V_Proj_mnist, 'tab:green', label = 'Proj. mnist')
plt.plot(range(len(Ierr_V_Proj_fashion)), Ierr_V_Proj_fashion, 'tab:cyan', label = 'Proj. fashion' )
# DynNet
plt.plot(range(len(Ierr_V_Dyn_mnist)), Ierr_V_Dyn_mnist, 'tab:brown', label = 'Dyn. mnist' )
plt.plot(range(len(Ierr_V_Dyn_fashion)), Ierr_V_Dyn_fashion, 'tab:gray', label = 'Dyn. fashion' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| V - \tilde V ||_F$')
plt.savefig('MNISTvsFashionMNIST_integrationerrorV100.png', bbox_inches='tight')
plt.show()