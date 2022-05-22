

from data import cifar10
from networks import DynTensorResNet
import optimisation
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import torch

N = 2500
V = 2500
batch_size = 5
# k must be less than 11 :-) for padding reasons
k = 9
transform ='tucker'

#data_cifar = cifar10(N, V, batch_size, k, transform) ## data object
data = cifar10(N, V, batch_size, k = k, transform= 'tucker' )
L = 10

net = DynTensorResNet(data, L)

print(net.net_structure)

torch.autograd.set_detect_anomaly(True)
_, acc_train, _, acc_val = optimisation.train(net,  max_epochs = 2)

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


print(net.orthogonality)

#print(net.rank_evolution)

print(net.get_integration_error)