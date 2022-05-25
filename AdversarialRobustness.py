from data import mnist, fashionMnist
from matplotlib import pyplot as plt
import torch
from networks import ResNet, ProjResNet, DynResNet
from optimisation import train
import torch.nn.functional as F
import numpy as np
from robustness import load_model, run_attack

N = 1500
V = 1500
batch_size = 5
max_epochs = 15
L = 10

## get data
#data_mnist = mnist( N, V, batch_size, k=28, transform='none')
#data_fashion = fashionMnist( N, V, batch_size, k=28, transform='none')

data_mnist_svd = mnist( N, V, batch_size, k=3, transform='svd')
#data_fashion_svd = fashionMnist( N, V, batch_size, k=3, transform='truncated svd')

## fetch networks
#path_mnist_resnet = "/Users/camillabalestrand/Masteroppgave/Models/mnist_resnet.pt"
#path_fashion_resnet = "/Users/camillabalestrand/Masteroppgave/Models/fashion_resnet.pt"

path_mnist_projnet = "/Users/camillabalestrand/Masteroppgave/Models/mnist_projnet.pt"
#path_fashion_projnet = "/Users/camillabalestrand/Masteroppgave/Models/fashion_projnet.pt"

#path_mnist_dynnet = "/Users/camillabalestrand/Masteroppgave/Models/mnist_dynnet.pt"
#path_fashion_dynnet = "/Users/camillabalestrand/Masteroppgave/Models/fashion_dynnet.pt"


model = load_model(DynResNet, data_mnist_svd, L, path_mnist_projnet)
epsilons = [0, .05, .1, .15, .2, .25, .3]

accuracies, examples = run_attack(model, epsilons)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex.reshape(28,28), cmap="gray")
plt.tight_layout()
plt.show()