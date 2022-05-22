from data import mnist
from matplotlib import pyplot as plt
import torch
from networks import ResNet
from optimisation import train
import numpy as np

N = 1500
V = 1500
batch_size = 5

plt.rcParams.update({
    "font.size":25})


### Orig ResNet w rank evolution vs Truncated X0 Resnet w rank evolution
## cannot monitor rank of cifar and svhn

# PREP DATA
data_mnist = mnist( N, V, batch_size, k=28, transform='none')
data_fashion = mnist( N, V, batch_size, k=28, transform='none')

k = 3
data_mnist_c = mnist( N, V, batch_size, k, transform='truncated svd')
data_fashion_c = mnist( N, V, batch_size, k, transform='truncated svd')

# CONSTRUCT NETWORK
net_mnist = ResNet(data_mnist, L=10, trainable_stepsize=True, d_hat='none')
net_fashion = ResNet(data_fashion, L=10, trainable_stepsize=True, d_hat='none')

net_mnist_c = ResNet(data_mnist_c, L=10, trainable_stepsize=True, d_hat='none')
net_fashion_c = ResNet(data_fashion_c, L=10, trainable_stepsize=True, d_hat='none')

# TRAIN NETWORK
torch.autograd.set_detect_anomaly(True)
_, acc_train_mnist, _, acc_val_mnist = train(net_mnist,  max_epochs = 15)
_, acc_train_fashion, _, acc_val_fashion = train(net_fashion,  max_epochs = 15)

_, acc_train_mnist_c, _, acc_val_mnist_c = train(net_mnist_c,  max_epochs = 15)
_, acc_train_fashion_c, _, acc_val_fashion_c = train(net_fashion_c,  max_epochs = 15)

# plot accurracy
plt.figure(figsize=(15, 8), dpi=80)
plt.plot(range(len(acc_train_mnist)), acc_train_mnist, 'tab:orange', label = 'mnist')
plt.plot(range(len(acc_val_mnist)), acc_val_mnist, 'tab:orange', linestyle='--')

plt.plot(range(len(acc_train_fashion)), acc_train_fashion, 'tab:purple', label = 'fashion' )
plt.plot(range(len(acc_val_fashion)), acc_val_fashion, 'tab:purple', linestyle='--')

plt.plot(range(len(acc_train_mnist_c)), acc_train_mnist_c, 'tab:green', label = 'mnist compr.')
plt.plot(range(len(acc_val_mnist_c)), acc_val_mnist_c, 'tab:green', linestyle='--')

plt.plot(range(len(acc_train_fashion_c)), acc_train_fashion_c, 'tab:cyan', label = 'fashion compr.' )
plt.plot(range(len(acc_val_fashion_c)), acc_val_fashion_c, 'tab:cyan', linestyle='--')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'Standard ResNet ')
plt.savefig('ResNetRankEvol_Accuracy.png', bbox_inches='tight')
plt.show()

### RANKS

ranks_mnist = net_mnist.rank_evolution
ranks_mnist_c = net_mnist_c.rank_evolution

ranks_fashion = net_fashion.rank_evolution
ranks_fashion_c = net_fashion_c.rank_evolution

plt.figure(figsize=(8, 6))
plt.plot(range(len(ranks_mnist)), np.around(ranks_mnist),  'tab:orange', linewidth=10, label = 'mnist')
plt.plot(range(len(ranks_fashion)), np.around(ranks_fashion), 'tab:purple', linewidth=7, label = 'fashion' )
plt.plot(range(len(ranks_mnist_c)), np.around(ranks_mnist_c), 'tab:green', linewidth=3,  label = 'mnist comp.')
plt.plot(range(len(ranks_fashion_c)), np.around(ranks_fashion_c), 'tab:cyan',  linewidth=1, label = 'fashion compr.' )
plt.legend()
plt.ylabel('Average rank')
plt.xlabel('Layer of network')
plt.title(r'Evolution of ranks in trained network ')
plt.savefig('EvolutionOfRanks.png')
plt.show()

print("\n")
print(max(acc_train_mnist))
print(max(acc_val_mnist))
print("\n")
print(max(acc_train_fashion))
print(max(acc_val_fashion))
print("\n")
print(max(acc_train_mnist_c))
print(max(acc_val_mnist_c))
print("\n")
print(max(acc_train_fashion_c))
print(max(acc_val_fashion_c))
print("\n")