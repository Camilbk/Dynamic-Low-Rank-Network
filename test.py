from data import mnist
from networks import DynResNet, ResNet, ProjResNet
import optimisation
import time
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import torch

N = 1500
V = 1500
batch_size = 5
k = 3
transform ='svd'

data_mnist = mnist(N, V, batch_size, k, transform) ## data object


print(data_mnist.toString)

net = DynResNet(data_mnist, L=10)
print(net.net_structure)

torch.autograd.set_detect_anomaly(True)
_, acc_train, _, acc_val = optimisation.train(net,  max_epochs = 15)



# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'after training'])
t.add_row(['training set', '%d %%' % acc_train[0], '%d %%' % acc_train[-1]])
t.add_row(['validation set', '%d %%' % acc_val[0], '%d %%' % acc_val[-1]])
print(t)

# plot accurracy
fig = plt.figure()
plt.plot(range(len(acc_train)), acc_train, label = 'training accuracy')
plt.plot(range(len(acc_val)), acc_val, label = 'validation accuracy')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'Network trained ')
plt.show()

print("Best training accuracy: ", round(max(acc_train), 2))
print("Best validation accuracy: ", round(max(acc_val), 2))

print("Stepsize: ", net.h)

print(net.orthogonality)
print(net.rank_evolution)
print(net.plot_integration_error)