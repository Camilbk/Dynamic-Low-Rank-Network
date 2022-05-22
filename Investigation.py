from data import mnist, fashionMnist, cifar10, svhn, from_tucker_decomposition
from networks import DynResNet, ResNet, ProjResNet
import optimisation
import time
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import torch
from torch.linalg import matrix_rank
import numpy as np

from tensorly.tucker_tensor import tucker_to_tensor
from tensorly import norm as tensor_norm
from tensorly.decomposition import tucker

plt.rcParams.update({
    "font.size":24})

N = 1500
V = 1500
k = 28
batch_size = 5
transform = 'none'

k = 28
data_mnist = mnist(N, V, batch_size, k, transform) ## data object
data_fashionMnist = fashionMnist(N, V, batch_size, k, transform) ## data object

k = 32
transform = 'grayscale'
data_cifar_g = cifar10(N, V, batch_size, k, transform) ## data object
data_svhn_g = svhn(N, V, batch_size, k, transform)


### RANKS AND GRAYSCALE RANK
plt.figure(figsize=(20, 8), dpi=80)
data_sets = [ data_fashionMnist, data_mnist, data_cifar_g, data_svhn_g]
for data in data_sets:
    name = str(type(data).__name__)
    data = data.all_data[0]
    n = data.shape[-1]
    data = data.unflatten(-1, (int(np.sqrt(n)),int(np.sqrt(n))))
    ranks = np.zeros(N)
    for i, image in enumerate(data):
        ranks[i] = matrix_rank(image)
    plt.plot(ranks, label = name)
plt.title('Rank of images in datasets')
plt.legend()
plt.xlabel("images")
plt.ylabel("rank")
plt.savefig('Rank_of_images_in_datasets', bbox_inches='tight')
#plt.show()

### ERROR IN TUCKER DECOMP
k = 32
transform = 'none'
data_cifar_orig = cifar10(N, V, batch_size, k, transform) ## data object
data_svhn_orig = svhn(N, V, batch_size, k, transform)

## ERROR in TUCKER DECOMP CIFAR10. using tucker rank r = [3, k, k]

train_cifar_orig = data_cifar_orig.all_data[0]
train_svhn_orig = data_svhn_orig.all_data[0]
k = 9
error_3 = np.zeros(N)
error_5 = np.zeros(N)
error_9 = np.zeros(N)
error_10 = np.zeros(N)
for i in range(N):
    orig_tensor = train_cifar_orig[i].unflatten(-1, (32, 32))
    k = 3
    compressed_3 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))
    k = 5
    compressed_5 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))
    k = 9
    compressed_9 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))
    k = 10
    compressed_10 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))

    error_3[i] = tensor_norm(orig_tensor.numpy() - compressed_3)
    error_5[i] = tensor_norm(orig_tensor.numpy() - compressed_5)
    error_9[i] = tensor_norm(orig_tensor.numpy() - compressed_9)
    error_10[i] = tensor_norm(orig_tensor.numpy() - compressed_10)

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(error_3, label='k = 3')
plt.plot(error_5, label='k = 5')
plt.plot(error_9, label='k = 9')
plt.plot(error_10, label='k = 10')
plt.title('Error in Tucker Decomposition CIFAR10')
plt.xlabel("images")
plt.ylabel(r"$|| Y - Y_k ||_F$")
plt.legend()
plt.savefig('Error_Tucker_CIFAR10', bbox_inches='tight')
#plt.show()


## ERROR in TUCKER DECOMP SVHN. using tucker rank r = [3, k, k]

train_svhn_orig = data_svhn_orig.all_data[0]
k = 9
error_3 = np.zeros(N)
error_5 = np.zeros(N)
error_9 = np.zeros(N)
error_10 = np.zeros(N)
for i in range(N):
    orig_tensor = train_svhn_orig[i].unflatten(-1, (32, 32))
    k = 3
    compressed_3 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))
    k = 5
    compressed_5 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))
    k = 9
    compressed_9 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))
    k = 10
    compressed_10 = tucker_to_tensor(tucker(orig_tensor.numpy(), rank=[3, k, k]))

    error_3[i] = tensor_norm(orig_tensor.numpy() - compressed_3)
    error_5[i] = tensor_norm(orig_tensor.numpy() - compressed_5)
    error_9[i] = tensor_norm(orig_tensor.numpy() - compressed_9)
    error_10[i] = tensor_norm(orig_tensor.numpy() - compressed_10)

plt.figure(figsize=(20, 8), dpi=80)
plt.plot(error_3, label = 'k = 3')
plt.plot(error_5, label = 'k = 5')
plt.plot(error_9, label = 'k = 9')
plt.plot(error_10, label = 'k = 10')
plt.title('Error in Tucker Decomposition SVHN')
plt.legend()
plt.xlabel("images")
plt.ylabel(r"$|| Y - Y_k ||_F$")
plt.savefig('Error_Tucker_SVHN', bbox_inches='tight')
#plt.show()


## SINGULAR VALUES
train_mnist = data_mnist.all_data[0].reshape((N,28,28))
train_fashion = data_fashionMnist.all_data[0].reshape((N,28,28))

train_cifar = data_cifar_g.all_data[0].unflatten(-1, (32,32))
train_svhn = data_svhn_g.all_data[0].unflatten(-1, (32,32))

from torch.linalg import svdvals

# data_sets = [train_mnist, train_fashion, train_cifar, train_svhn]

fig, ax = plt.subplots(2, 2, figsize=(24, 15), dpi=80)

for data in train_mnist:
    name = str(type(data).__name__)
    sigmas = svdvals(data)
    ax[0, 0].plot(sigmas, "o", label=name)
for data in train_fashion:
    name = str(type(data).__name__)
    sigmas = svdvals(data)
    ax[0, 1].plot(sigmas, "o", label=name)
for data in train_cifar:
    name = str(type(data).__name__)
    sigmas = svdvals(data)
    ax[1, 0].plot(sigmas, "o", label=name)
for data in train_svhn:
    name = str(type(data).__name__)
    sigmas = svdvals(data)
    ax[1, 1].plot(sigmas, "o", label=name)

ax[0, 0].set_title('Singular values MNIST')
ax[0, 1].set_title('Singular values FashionMNIST')
ax[1, 0].set_title('Singular values CIFAR10 grayscale')
ax[1, 1].set_title('Singular values SVHN grayscale')
plt.xlabel("Singular values")
plt.ylabel("Value")
plt.savefig('SingularValues')
plt.show()