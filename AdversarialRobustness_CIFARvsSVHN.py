from data import cifar10, svhn
from matplotlib import pyplot as plt
from networks import ResNet, ProjTensorResNet, DynTensorResNet
import numpy as np
from robustness import load_model, run_attack, plot_adversarial_examples

N = 3500
V = 3500
batch_size = 5
max_epochs = 15
L = 10

epsilons = [0, .05, .1, .15, .2, .25, .3]

#CIFAR

## get data
data_cifar = cifar10( N, V, batch_size, k=28, transform='none')
data_cifar_tucker = cifar10( N, V, batch_size, k=3, transform='tucker')

## fetch networks
path_mnist_resnet = "../../Models/cifar_resnet.pt"
path_mnist_projnet = "../../Models/cifar_projnet.pt"
path_mnist_dynnet = "../../Models/cifar_dynnet.pt"

model = load_model(ResNet, data_cifar, L, path_mnist_resnet)
accuracies_resnet, examples_resnet = run_attack(model, epsilons)

model = load_model(ProjTensorResNet, data_cifar_tucker, L, path_mnist_projnet)
accuracies_projnet, examples_projnet = run_attack(model, epsilons)

model = load_model(DynTensorResNet, data_cifar_tucker, L, path_mnist_dynnet)
accuracies_dynnet, examples_dynnet = run_attack(model, epsilons)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies_resnet, ".-", label = "ResNet")
plt.plot(epsilons, accuracies_projnet, ".-", label = "ProjNet")
plt.plot(epsilons, accuracies_dynnet, ".-", label = "DynNet")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("CIFAR10 Adversarial FGS Attack")
plt.xlabel("Epsilon")
plt.legend()
plt.ylabel("Accuracy")
plt.savefig('CIFAR_AdversarialAttack.png', bbox_inches='tight')
plt.show()

plot_adversarial_examples(epsilons, examples_resnet)
plt.savefig('CIFAR_AdversaialExamples_ResNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_projnet, k = 3, transform='svd')
plt.savefig('CIFAR_AdversaialExamples_ProjNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_dynnet, k = 3, transform='svd')
plt.savefig('CIFAR_AdversaialExamples_DynNet.png', bbox_inches='tight')


## SVHN

data_svhn = svhn( N, V, batch_size, k=28, transform='none')
data_svhn_tucker = svhn( N, V, batch_size, k=3, transform='tucker')

path_svhn_resnet = "../../Models/svhn_resnet.pt"
path_svhn_projnet = "../../Models/svhn_projnet.pt"
path_svhn_dynnet = "../../Models/svhn_dynnet.pt"

model = load_model(ResNet, data_svhn, L, path_svhn_resnet)
accuracies_resnet, examples_resnet = run_attack(model, epsilons)

model = load_model(ProjTensorResNet, data_svhn_tucker, L, path_svhn_projnet)
accuracies_projnet, examples_projnet = run_attack(model, epsilons)

model = load_model(DynTensorResNet, data_svhn_tucker, L, path_svhn_dynnet)
accuracies_dynnet, examples_dynnet = run_attack(model, epsilons)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies_resnet, ".-", label = "ResNet")
plt.plot(epsilons, accuracies_projnet, ".-", label = "ProjNet")
plt.plot(epsilons, accuracies_dynnet, ".-", label = "DynNet")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("SVHN Adversarial FGS Attack")
plt.xlabel("Epsilon")
plt.legend()
plt.ylabel("Accuracy")
plt.savefig('SVHN_AdversarialAttack.png', bbox_inches='tight')
plt.show()


plot_adversarial_examples(epsilons, examples_resnet)
plt.savefig('SVHN_AdversaialExamples_ResNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_projnet, k = 3, transform='svd')
plt.savefig('SVHN_AdversaialExamples_ProjNet.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_dynnet, k = 3, transform='svd')
plt.savefig('SVHN_AdversaialExamples_DynNet.png', bbox_inches='tight')
