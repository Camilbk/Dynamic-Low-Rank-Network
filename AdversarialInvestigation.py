from data import mnist, fashionMnist
from matplotlib import pyplot as plt
import torch
from networks import ResNet
from optimisation import train
import numpy as np
from robustness import run_attack, plot_adversarial_examples

epsilons = [0, .05, .1, .15, .2, .25, .3]

N = 5000
V = 1500
batch_size = 5
max_epochs = 15
k = 3

plt.rcParams.update({
    "font.size":30})


### Orig ResNet w rank evolution vs Truncated X0 Resnet w rank evolution
## FashionMNIST

# PREP DATA
data = fashionMnist( N, V, batch_size, k=28, transform='none')
data_c = fashionMnist( N, V, batch_size, k, transform='truncated svd')

net = ResNet(data, L=100, trainable_stepsize=True, d_hat='none') # standard
net_c = ResNet(data_c, L=100, trainable_stepsize=True, d_hat='none') # compressed initial cond

# problematic alg. fun first
net_r = ResNet(data_c, L=100, trainable_stepsize=True, d_hat='none', perform_svd=True) # rank restricted
_, acc_train_r100, _, acc_val_r100 = train(net_r,  max_epochs = 5)
net_r = ResNet(data_c, L=10, trainable_stepsize=True, d_hat='none', perform_svd=True) # rank restricted
_, acc_train_r, _, acc_val_r = train(net_r,  max_epochs = max_epochs)

_, acc_train_100, _, acc_val_100 = train(net,  max_epochs = max_epochs)
_, acc_train_c100, _, acc_val_c100 = train(net_c,  max_epochs = max_epochs)

net.eval()
net_c.eval()
net_r.eval()
data = fashionMnist( N=1500, V=1500, batch_size=5, k=28, transform='none')
data_c = fashionMnist( N=1500, V=1500, batch_size=5, k=3, transform='truncated svd')
accuracies_100, _ = run_attack(net, epsilons)
accuracies_100_c, _ = run_attack(net_c, epsilons)
accuracies_100_r, _ = run_attack(net_r, epsilons)

net = ResNet(data, L=10, trainable_stepsize=True, d_hat='none') # standard
net_c = ResNet(data_c, L=10, trainable_stepsize=True, d_hat='none') # compressed initial cond


_, acc_train, _, acc_val = train(net,  max_epochs = max_epochs)
_, acc_train_c, _, acc_val_c = train(net_c,  max_epochs = max_epochs)


net.eval()
net_c.eval()
net_r.eval()
accuracies, _ = run_attack(net, epsilons)
accuracies_c, examples_c = run_attack(net_c, epsilons)
accuracies_r, examples_r = run_attack(net_r, epsilons)

# plot accurracy
plt.figure(figsize=(15, 8), dpi=80)
plt.plot(range(len(acc_train)), acc_train, 'tab:orange', linewidth=3,label = 'fashion-10')
plt.plot(range(len(acc_val)), acc_val, 'tab:orange',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_c)), acc_train_c, 'tab:green',linewidth=3, label = 'fashion-10 compr.')
plt.plot(range(len(acc_val_c)), acc_val_c, 'tab:green',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_r)), acc_train_r, 'tab:purple',linewidth=3, label = 'fashion-10 restr.')
plt.plot(range(len(acc_val_r)), acc_val_r, 'tab:purple',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_100)), acc_train_100, 'tab:cyan', linewidth=3,label = 'fashion-100')
plt.plot(range(len(acc_val_100)), acc_val_100, 'tab:cyan',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_c100)), acc_train_c100, 'tab:brown',linewidth=3, label = 'fashion-100 compr.')
plt.plot(range(len(acc_val_c100)), acc_val_c100, 'tab:brown',linewidth=3, linestyle='--')

plt.plot(range(len(acc_train_r100)), acc_train_r100, 'tab:gray',linewidth=3, label = 'fashion-100 restr.')
plt.plot(range(len(acc_val_r100)), acc_val_r100, 'tab:gray',linewidth=3, linestyle='--')


plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'Convergence of ResNets on FashionMNIST  ')
plt.savefig('ResNetRankEvol_Accuracy_fashion_compressed.png', bbox_inches='tight')
plt.show()


plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, ".-", label = "fashion-10")
plt.plot(epsilons, accuracies_c, ".-", label = "fashion-10 compr.")
plt.plot(epsilons, accuracies_r, ".-", label = "fashion-10 restr.")

plt.plot(epsilons, accuracies_100, ".-", label = "fashion-100")
plt.plot(epsilons, accuracies_100_c, ".-", label = "fashion-100 compr.")
plt.plot(epsilons, accuracies_100_r, ".-", label = "fashion-100 restr.")

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("FashionMNIST Adversarial FGS Attack")
plt.xlabel("Epsilon")
plt.legend()
plt.ylabel("Accuracy")
plt.savefig('FashionMNIST_AdversarialAttack_compressions.png', bbox_inches='tight')
plt.show()

plot_adversarial_examples(epsilons, examples_c)
plt.savefig('FashionMNIST_AdversaialExamples_c.png', bbox_inches='tight')
plot_adversarial_examples(epsilons, examples_r)
plt.savefig('FashionMNIST_AdversaialExamples_r.png', bbox_inches='tight')
plt.show()