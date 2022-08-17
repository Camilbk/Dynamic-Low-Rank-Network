from data import cifar10, svhn
from networks import ResNet, ProjResNet, DynResNet
from robustness import load_model, run_attack, plot_adversarial_examples
from matplotlib import pyplot as plt
import torch
from networks import ResNet, ProjTensorResNet, DynTensorResNet

N = 1500
V = 1500
batch_size = 5
max_epochs = 15


#CIFAR
L = 10

## get data
#data = cifar10( N, V, batch_size, k=28, transform='none')
data_cifar = cifar10( N, V, batch_size, k=9, transform='tucker')
data_svhn = svhn( N, V, batch_size, k=9, transform='tucker')

## fetch networks
# L = 10

plt.rcParams.update({
    "font.size":40})

path = "/Users/camillabalestrand/Masteroppgave/Models/cifar_dynnet10.pt"
model = load_model(DynTensorResNet, data_cifar, L, path)
s, err_U1_Dyn_cifar, err_U2_Dyn_cifar, err_U3_Dyn_cifar = model.orthogonality
print(s)
path = "/Users/camillabalestrand/Masteroppgave/Models/svhn_dynnet10.pt"
model = load_model(DynTensorResNet, data_svhn, L, path)
s, err_U1_Dyn_svhn, err_U2_Dyn_svhn, err_U3_Dyn_svhn = model.orthogonality
print(s)
print("\n")

path = "/Users/camillabalestrand/Masteroppgave/Models/cifar_projnet10.pt"
model = load_model(ProjTensorResNet, data_cifar, L, path)
s, err_U1_Proj_cifar, err_U2_Proj_cifar, err_U3_Proj_cifar = model.orthogonality
print(s)
path = "/Users/camillabalestrand/Masteroppgave/Models/svhn_projnet10.pt"
model = load_model(ProjTensorResNet, data_svhn, L, path)
s, err_U1_Proj_svhn, err_U2_Proj_svhn, err_U3_Proj_svhn = model.orthogonality
print(s)
print("\n")


plt.figure(figsize=(14, 8), dpi=80)
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
plt.savefig('CIFARvsSVHN_orthogonalityU1_%i.png' %L, bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 8), dpi=80)
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
plt.savefig('CIFARvsSVHN_orthogonalityU2_%i.png' %L, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 8), dpi=80)
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
plt.savefig('CIFARvsSVHN_orthogonalityU3_%i.png' %L, bbox_inches='tight')
plt.show()









