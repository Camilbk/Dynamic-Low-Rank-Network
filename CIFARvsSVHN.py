from data import cifar10, svhn
from matplotlib import pyplot as plt
import torch
from networks import ResNet, ProjTensorResNet, DynTensorResNet
from optimisation import train
import numpy as np

N = 10000
<<<<<<< HEAD
V = 5000
batch_size = 15

L = 10
max_epochs = 40
=======
V = 10000
batch_size = 32

L = 10
max_epochs = 50
>>>>>>> bfa1792d3f6dfe7a9ba47a8b5d9b7ec11f171804

k = 9

plt.rcParams.update({
    "font.size":25})



#### DYNAMIC LOW-RANK NET
# DATA
data_cifar_tucker = cifar10( N, V, batch_size, k=9, transform='tucker')
data_svhn_tucker = svhn( N, V, batch_size, k=9, transform='tucker')
# NETWORK
# CONSTRUCT NETWORK
DynResNet_cifar = DynTensorResNet(data_cifar_tucker, L, d_hat='none', use_cayley=False)
DynResNet_svhn = DynTensorResNet(data_svhn_tucker, L, d_hat='none', use_cayley=False)
# TRAIN NETWORK
torch.autograd.set_detect_anomaly(True)
_, Dyn_acc_train_cifar, _, Dyn_acc_val_cifar = train(DynResNet_cifar,  max_epochs = max_epochs)
print("DynTensorResNet")
print("cifar")
print(max(Dyn_acc_train_cifar))
print(max(Dyn_acc_val_cifar))
_, Dyn_acc_train_svhn , _, Dyn_acc_val_svhn  = train(DynResNet_svhn,  max_epochs = max_epochs)
print("svhn ")
print(max(Dyn_acc_train_svhn ))
print(max(Dyn_acc_val_svhn))
print("\n")
#save models
PATH_cifar = "../../Models/cifar_dynnet.pt"
PATH_svhn = "../../Models/svhn_dynnet.pt"
torch.save(DynResNet_cifar.best_state, PATH_cifar)
torch.save(DynResNet_svhn.best_state, PATH_svhn)

#### STANDARD TENSOR RESNET
# DATA
data_cifar = cifar10( N, V, batch_size, k=32, transform='none')
data_svhn = svhn( N, V, batch_size, k=32, transform='none')
# NETWORK
# CONSTRUCT NETWORK
net_cifar = ResNet(data_cifar, L=L, trainable_stepsize=True, d_hat='none')
net_svhn = ResNet(data_svhn, L=L, trainable_stepsize=True, d_hat='none')
# TRAIN NETWORK
torch.autograd.set_detect_anomaly(True)
_, acc_train_cifar, _, acc_val_cifar = train(net_cifar,  max_epochs = max_epochs)
print("ResNet")
print("cifar10")
print(max(acc_train_cifar))
print(max(acc_val_cifar))
_, acc_train_svhn, _, acc_val_svhn = train(net_svhn,  max_epochs = max_epochs)
print("svhn")
print(max(acc_train_svhn))
print(max(acc_val_svhn))
#save models
PATH_cifar = "../../Models/cifar_resnet.pt"
PATH_svhn = "../../Models/svhn_resnet.pt"
torch.save(net_cifar.best_state, PATH_cifar)
torch.save(net_svhn.best_state, PATH_svhn)

#### PROJECTION TENSOR RESNET
# DATA
#data_cifar_tucker = cifar10( N, V, batch_size, k=k, transform='tucker')
#data_svhn_tucker = svhn( N, V, batch_size, k=k, transform='tucker')
# NETWORK
# CONSTRUCT NETWORK
ProjNet_cifar = ProjTensorResNet(data_cifar_tucker, L, trainable_stepsize=True, d_hat='none')
ProjNet_svhn = ProjTensorResNet(data_svhn_tucker, L, trainable_stepsize=True, d_hat='none')
# TRAIN NETWORK
torch.autograd.set_detect_anomaly(True)
_, Proj_acc_train_cifar, _, Proj_acc_val_cifar = train(ProjNet_cifar,  max_epochs = max_epochs)
_, Proj_acc_train_svhn, _, Proj_acc_val_svhn = train(ProjNet_svhn,  max_epochs = max_epochs)
#save models
PATH_cifar = "../../Models/cifar_projnet.pt"
PATH_svhn = "../../Models/svhn_projnet.pt"
torch.save(ProjNet_cifar.best_state, PATH_cifar)
torch.save(ProjNet_svhn.best_state, PATH_svhn)


print("ResNet")
print("cifar10")
print(max(acc_train_cifar))
print(max(acc_val_cifar))
print("svhn")
print(max(acc_train_svhn))
print(max(acc_val_svhn))
print("ProjTensoResNet")
print("cifar10")
print(max(Proj_acc_train_cifar))
print(max(Proj_acc_val_cifar))
print("svhn")
print(max(Proj_acc_train_svhn))
print(max(Proj_acc_val_svhn))
print("DynTensorResNet")
print("cifar")
print(max(Dyn_acc_train_cifar))
print(max(Dyn_acc_val_cifar))
print("svhn ")
print(max(Dyn_acc_train_svhn ))
print(max(Dyn_acc_val_svhn))
print("\n")


# plot accurracy

plt.figure(figsize=(12, 8), dpi=80)
### ResNet
plt.plot(range(len(acc_train_cifar)), acc_train_cifar, 'tab:orange', label = 'cifar10')
plt.plot(range(len(acc_val_cifar)), acc_val_cifar, 'tab:orange', linestyle='--')

plt.plot(range(len(acc_train_svhn)), acc_train_svhn, 'tab:purple', label = 'svhn' )
plt.plot(range(len(acc_val_svhn)), acc_val_svhn, 'tab:purple', linestyle='--')

# ProjNet
plt.plot(range(len(Proj_acc_train_cifar)), Proj_acc_train_cifar, 'tab:green', label = 'Proj. cifar10')
plt.plot(range(len(Proj_acc_val_cifar)), Proj_acc_val_cifar, 'tab:green', linestyle='--')

plt.plot(range(len(Proj_acc_train_svhn)), Proj_acc_train_svhn, 'tab:cyan', label = 'Proj. svhn' )
plt.plot(range(len(Proj_acc_val_svhn)), Proj_acc_val_svhn, 'tab:cyan', linestyle='--')

# DynNet
plt.plot(range(len(Dyn_acc_train_cifar)), Dyn_acc_train_cifar, 'tab:brown', label = 'Dyn. cifar10' )
plt.plot(range(len(Dyn_acc_val_cifar)), Dyn_acc_val_cifar, 'tab:brown', linestyle='--')

plt.plot(range(len(Dyn_acc_train_svhn)), Dyn_acc_train_svhn, 'tab:gray', label = 'Dyn. svhn' )
plt.plot(range(len(Dyn_acc_val_svhn)), Dyn_acc_val_svhn, 'tab:gray', linestyle='--')

plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'CIFAR10 vs SVHN ')
plt.savefig('CIFARvsSVHN_accuracy.png', bbox_inches='tight')
plt.show()



# plot orthogonality

s, err_U1_Proj_cifar, err_U2_Proj_cifar, err_U3_Proj_cifar = ProjNet_cifar.orthogonality
print(s)
s, err_U1_Proj_svhn, err_U2_Proj_svhn, err_U3_Proj_svhn = ProjNet_svhn.orthogonality
print(s)
print("\n")

s, err_U1_Dyn_cifar, err_U2_Dyn_cifar, err_U3_Dyn_cifar  = DynResNet_cifar.orthogonality
print(s)
s, err_U1_Dyn_svhn, err_U2_Dyn_svhn, err_U3_Dyn_svhn = DynResNet_svhn.orthogonality
print(s)

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_U1_Proj_cifar)), err_U1_Proj_cifar, 'tab:green', label = 'Proj. cifar')
plt.plot(range(len(err_U1_Proj_svhn)), err_U1_Proj_svhn, 'tab:cyan', label = 'Proj. svhn' )
# DynNet
plt.plot(range(len(err_U1_Dyn_cifar)), err_U1_Dyn_cifar, 'tab:brown', label = 'Dyn. cifar' )
plt.plot(range(len(err_U1_Dyn_svhn)), err_U1_Dyn_svhn, 'tab:gray', label = 'Dyn. svhn' )

plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - U_1^T U_1 ||_F$')
plt.savefig('CIFARvsSVHN_orthogonalityU1.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_U2_Proj_cifar)), err_U2_Proj_cifar, 'tab:green', label = 'Proj. cifar')
plt.plot(range(len(err_U2_Proj_svhn)), err_U2_Proj_svhn, 'tab:cyan', label = 'Proj. svhn' )
# DynNet
plt.plot(range(len(err_U2_Dyn_cifar)), err_U2_Dyn_cifar, 'tab:brown', label = 'Dyn. cifar' )
plt.plot(range(len(err_U2_Dyn_svhn)), err_U2_Dyn_svhn, 'tab:gray', label = 'Dyn. svhn' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - U_2^T U_2 ||_F$')
plt.savefig('CIFARvsSVHN_orthogonalityU2.png', bbox_inches='tight')
plt.show()


plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(err_U3_Proj_cifar)), err_U3_Proj_cifar, 'tab:green', label = 'Proj. cifar')
plt.plot(range(len(err_U3_Proj_svhn)), err_U3_Proj_svhn, 'tab:cyan', label = 'Proj. svhn' )
# DynNet
plt.plot(range(len(err_U3_Dyn_cifar)), err_U3_Dyn_cifar, 'tab:brown', label = 'Dyn. cifar' )
plt.plot(range(len(err_U3_Dyn_svhn)), err_U3_Dyn_svhn, 'tab:gray', label = 'Dyn. svhn' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| I - U_3^T U_3 ||_F$')
plt.savefig('CIFARvsSVHN_orthogonalityU3.png', bbox_inches='tight')
plt.show()



# integration error ..  ?



Ierr_U1_Proj_cifar, Ierr_U2_Proj_cifar, Ierr_U3_Proj_cifar = ProjNet_cifar.get_integration_error
Ierr_U1_Proj_svhn, Ierr_U2_Proj_svhn, Ierr_U3_Proj_svhn = ProjNet_svhn.get_integration_error

Ierr_U1_Dyn_cifar, Ierr_U2_Dyn_cifar, Ierr_U3_Dyn_cifar = DynResNet_cifar.get_integration_error
Ierr_U1_Dyn_svhn, Ierr_U2_Dyn_svhn, Ierr_U3_Dyn_svhn = DynResNet_svhn.get_integration_error

print("integration error U1 Proj CIFAR: ", Ierr_U1_Proj_cifar)
print("integration error U2 Proj CIFAR: ", Ierr_U2_Proj_cifar)
print("integration error U3 Proj CIFAR: ", Ierr_U3_Proj_cifar)

print("integration error U1 Proj SVHN: ", Ierr_U1_Proj_svhn)
print("integration error U2 Proj SVHN: ", Ierr_U2_Proj_svhn)
print("integration error U3 Proj SVHN: ", Ierr_U3_Proj_svhn)

print("integration error U1 Dyn CIFAR: ", Ierr_U1_Dyn_cifar)
print("integration error U2 Dyn CIFAR: ", Ierr_U2_Dyn_cifar)
print("integration error U3 Dyn CIFAR: ", Ierr_U3_Dyn_cifar)

print("integration error U1 Dyn SVHN: ", Ierr_U1_Dyn_svhn)
print("integration error U2 Dyn SVHN: ", Ierr_U2_Dyn_svhn)
print("integration error U3 Dyn SVHN: ", Ierr_U3_Dyn_svhn)

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(Ierr_U1_Proj_cifar)), Ierr_U1_Proj_cifar, 'tab:green', label = 'Proj. cifar')
plt.plot(range(len(Ierr_U1_Proj_svhn)), Ierr_U1_Proj_svhn, 'tab:cyan', label = 'Proj. svhn' )
# DynNet
plt.plot(range(len(Ierr_U1_Dyn_cifar)), Ierr_U1_Dyn_cifar, 'tab:brown', label = 'Dyn. cifar' )
plt.plot(range(len(Ierr_U1_Dyn_svhn)), Ierr_U1_Dyn_svhn, 'tab:gray', label = 'Dyn. svhn' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| U_1 - \tilde U_1 ||_F$')
plt.savefig('CIFARvsSVHN_integrationerrorU1.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(Ierr_U2_Proj_cifar)), Ierr_U2_Proj_cifar, 'tab:green', label = 'Proj. cifar')
plt.plot(range(len(Ierr_U2_Proj_svhn)), Ierr_U2_Proj_svhn, 'tab:cyan', label = 'Proj. svhn' )
# DynNet
plt.plot(range(len(Ierr_U2_Dyn_cifar)), Ierr_U2_Dyn_cifar, 'tab:brown', label = 'Dyn. cifar' )
plt.plot(range(len(Ierr_U2_Dyn_svhn)), Ierr_U2_Dyn_svhn, 'tab:gray', label = 'Dyn. svhn' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| U_2 - \tilde U_2 ||_F$')
plt.savefig('CIFARvsSVHN_integrationerrorU2.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
# ProjNet
plt.plot(range(len(Ierr_U3_Proj_cifar)), Ierr_U3_Proj_cifar, 'tab:green', label = 'Proj. cifar')
plt.plot(range(len(Ierr_U3_Proj_svhn)), Ierr_U3_Proj_svhn, 'tab:cyan', label = 'Proj. svhn' )
# DynNet
plt.plot(range(len(Ierr_U3_Dyn_cifar)), Ierr_U3_Dyn_cifar, 'tab:brown', label = 'Dyn. cifar' )
plt.plot(range(len(Ierr_U3_Dyn_svhn)), Ierr_U3_Dyn_svhn, 'tab:gray', label = 'Dyn. svhn' )
plt.legend()
plt.ylabel('error')
plt.xlabel('layers')
plt.title(r' $|| U_3 - \tilde U_3 ||_F$')
plt.savefig('CIFARvsSVHN_integrationerrorU3.png', bbox_inches='tight')
plt.show()
