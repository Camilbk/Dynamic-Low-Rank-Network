from data import mnist, fashionMnist
from matplotlib import pyplot as plt
import torch
from networks import ResNet, ProjResNet, DynResNet
from optimisation import train
import torch.nn.functional as F
import numpy as np
from robustness import load_model, run_attack, plot_adversarial_examples

N = 1500
V = 1500
batch_size = 5
max_epochs = 15
L = 10

epsilons = [0, .05, .1, .15, .2, .25, .3]

#MNIST

## get data
data_mnist = mnist( N, V, batch_size, k=28, transform='none')
data_mnist_svd = mnist( N, V, batch_size, k=3, transform='svd')

## fetch networks
path_mnist_resnet = "/Users/camillabalestrand/Masteroppgave/Models/mnist_resnet.pt"
path_mnist_projnet = "/Users/camillabalestrand/Masteroppgave/Models/mnist_projnet.pt"
path_mnist_dynnet = "/Users/camillabalestrand/Masteroppgave/Models/mnist_dynnet.pt"

model = load_model(ResNet, data_mnist, L, path_mnist_resnet)
#accuracies_resnet, examples_resnet = run_attack(model, epsilons)

model = load_model(ProjResNet, data_mnist_svd, L, path_mnist_projnet)
accuracies_projnet, examples_projnet = run_attack(model, epsilons)

model = load_model(DynResNet, data_mnist_svd, L, path_mnist_dynnet)
#accuracies_dynnet, examples_dynnet = run_attack(model, epsilons)

plt.figure(figsize=(5,5))
#plt.plot(epsilons, accuracies_resnet, ".-", label = "ResNet")
plt.plot(epsilons, accuracies_projnet, ".-", label = "ProjNet")
#plt.plot(epsilons, accuracies_dynnet, ".-", label = "DynNet")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("MNIST Adversarial FGS Attack")
plt.xlabel("Epsilon")
plt.legend()
plt.ylabel("Accuracy")
plt.savefig('MNIST_AdversarialAttack.png', bbox_inches='tight')
plt.show()

#plot_adversarial_examples(epsilons, examples_resnet)
#plt.savefig('MNIST_AdversaialExamples_ResNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_projnet, k = 3, transform='svd')
plt.savefig('MNIST_AdversaialExamples_ProjNet.png', bbox_inches='tight')
#plot_adversarial_examples(epsilons, examples_dynnet, k = 3, transform='svd')
#plt.savefig('MNIST_AdversaialExamples_DynNet.png', bbox_inches='tight')


## FashionMNIST

data_fashion = fashionMnist( N, V, batch_size, k=28, transform='none')
data_fashion_svd = fashionMnist( N, V, batch_size, k=3, transform='svd')

path_fashion_resnet = "/Users/camillabalestrand/Masteroppgave/Models/fashion_resnet.pt"
path_fashion_projnet = "/Users/camillabalestrand/Masteroppgave/Models/fashion_projnet.pt"
path_fashion_dynnet = "/Users/camillabalestrand/Masteroppgave/Models/fashion_dynnet.pt"

model = load_model(ResNet, data_fashion, L, path_fashion_resnet)
accuracies_resnet, examples_resnet = run_attack(model, epsilons)

model = load_model(ProjResNet, data_fashion_svd, L, path_fashion_projnet)
accuracies_projnet, examples_projnet = run_attack(model, epsilons)

model = load_model(DynResNet, data_fashion_svd, L, path_fashion_dynnet)
accuracies_dynnet, examples_dynnet = run_attack(model, epsilons)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies_resnet, ".-", label = "ResNet")
plt.plot(epsilons, accuracies_projnet, ".-", label = "ProjNet")
plt.plot(epsilons, accuracies_dynnet, ".-", label = "DynNet")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("FashionMNIST Adversarial FGS Attack")
plt.xlabel("Epsilon")
plt.legend()
plt.ylabel("Accuracy")
plt.savefig('FashionMNIST_AdversarialAttack.png', bbox_inches='tight')
plt.show()


plot_adversarial_examples(epsilons, examples_resnet)
plt.savefig('FashionMNIST_AdversaialExamples_ResNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_projnet, k = 3, transform='svd')
plt.savefig('FashionMNIST_AdversaialExamples_ProjNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_dynnet, k = 3, transform='svd')
plt.savefig('FashionMNIST_AdversaialExamples_DynNet.png', bbox_inches='tight')
