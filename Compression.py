from data import cifar10, svhn
from matplotlib import pyplot as plt
import torch
from networks import ResNet
from optimisation import train
import numpy as np

N = 150
V = 150
batch_size = 5
max_epochs= 2

plt.rcParams.update({
    "font.size":30})


### Orig ResNet w rank evolution vs Truncated X0 Resnet w rank evolution
## cannot monitor rank of cifar and svhn

# PREP DATA
data_cifar = cifar10( N, V, batch_size, k=32, transform='none')
data_svhn = svhn( N, V, batch_size, k=32, transform='none')

k = 9
data_cifar_c = cifar10( N, V, batch_size, k, transform='truncated tucker')
data_svhn_c = svhn( N, V, batch_size, k, transform='truncated tucker')

torch.autograd.set_detect_anomaly(True)
# CONSTRUCT NETWORK & TRAIN
net = ResNet(data_cifar, L=10, trainable_stepsize=True, d_hat='none')
_, acc_train_cifar, _, acc_val_cifar = train(net,  max_epochs = max_epochs )

net = ResNet(data_svhn, L=10, trainable_stepsize=True, d_hat='none')
_, acc_train_svhn, _, acc_val_svhn = train(net,  max_epochs = max_epochs)

net = ResNet(data_cifar_c, L=10, trainable_stepsize=True, d_hat='none')
_, acc_train_cifar_c , _, acc_val_cifar_c  = train(net,  max_epochs = max_epochs)

net = ResNet(data_svhn_c, L=10, trainable_stepsize=True, d_hat='none')
_, acc_train_svhn_c, _, acc_val_svhn_c= train(net,  max_epochs = max_epochs)

# plot accurracy
plt.figure(figsize=(15, 8), dpi=80)
plt.plot(range(len(acc_train_cifar)), acc_train_cifar, 'tab:orange', linewidth=3,label = 'cifar10')
plt.plot(range(len(acc_val_cifar)), acc_val_cifar, 'tab:orange',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_svhn)), acc_train_svhn, 'tab:purple', linewidth=3,label = 'svhn' )
plt.plot(range(len(acc_val_svhn)), acc_val_svhn, 'tab:purple',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_cifar_c)), acc_train_cifar_c, 'tab:green',linewidth=3, label = 'cifar10 compr.')
plt.plot(range(len(acc_val_cifar_c)), acc_val_cifar_c, 'tab:green',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_svhn_c)), acc_train_svhn_c, 'tab:cyan',linewidth=3, label = 'svhn compr.' )
plt.plot(range(len(acc_val_svhn_c)), acc_val_svhn_c, 'tab:cyan',linewidth=3, linestyle='--')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'Standard ResNet ')
plt.savefig('ResNetRankEvol_Accuracy_CIFAR_SVHN.png', bbox_inches='tight')
plt.show()



print("\n")
print(max(acc_train_cifar))
print(max(acc_val_cifar))
print("\n")
print(max(acc_train_svhn))
print(max(acc_val_svhn))
print("\n")
print(max(acc_train_cifar_c))
print(max(acc_val_cifar_c))
print("\n")
print(max(acc_train_svhn_c))
print(max(acc_val_svhn_c))
print("\n")