import torch
from data import mnist
from networks import ResNet, SVDResNet
from optimisation import train
from prettytable import PrettyTable
from matplotlib import pyplot as plt

# data sizes and batch size for sgd
N = 1500
V = 1500
batch_size = 5

# layers of network
L = 10
# epochs to run
epochs = 1

data_tsvd3 = mnist(N, V, batch_size, k=3, transform = 'truncated svd') # data object
data_tsvd28 = mnist(N, V, batch_size, k=28, transform = 'truncated svd') # data object
# Test how original performs

# Test how SVD on every time step performs ( truncated )
# k = 28
"""
data_orig = mnist(N, V, batch_size, k=28, transform = 'none') ## data object
net_orig = ResNet(data_orig, L)

torch.autograd.set_detect_anomaly(True)
_, acc_train_orig, _, acc_val_orig = train(net_orig,  max_epochs = epochs)
print(net_orig.h)

# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'during training ', 'after training'])
t.add_row(['training set', '%d %%' % acc_train_orig[0],  round(max(acc_train_orig),2), '%d %%' % acc_train_orig[-1]])
t.add_row(['validation set', '%d %%' % acc_val_orig[0],  round(max(acc_val_orig),2), '%d %%' % acc_val_orig[-1]])
print(t)


## Test hor Initial Value SVD performs

#k = 28

net_iv28 = ResNet(data_tsvd28, L)
torch.autograd.set_detect_anomaly(True)
_, acc_train_iv28, _, acc_val_iv28 = train(net_iv28,  max_epochs = epochs)
print(net_iv28.h)

# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'during training ', 'after training'])
t.add_row(['training set', '%d %%' % acc_train_iv28[0],  round(max(acc_train_iv28),2), '%d %%' % acc_train_iv28[-1]])
t.add_row(['validation set', '%d %%' % acc_val_iv28[0],  round(max(acc_val_iv28),2), '%d %%' % acc_val_iv28[-1]])
print(t)

# k = 3

net_iv3 = ResNet(data_tsvd3, L)
torch.autograd.set_detect_anomaly(True)
_, acc_train_iv3, _, acc_val_iv3 = train(net_iv3,  max_epochs = epochs)
print(net_iv3.h)

# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'during training ', 'after training'])
t.add_row(['training set', '%d %%' % acc_train_iv3[0],  round(max(acc_train_iv3),2), '%d %%' % acc_train_iv3[-1]])
t.add_row(['validation set', '%d %%' % acc_val_iv3[0],  round(max(acc_val_iv3),2), '%d %%' % acc_val_iv3[-1]])
print(t)


net_svd28 = SVDResNet(data_tsvd28, L)
torch.autograd.set_detect_anomaly(True)
_, acc_train_28, _, acc_val_28 = train(net_svd28,  max_epochs = epochs)
print(net_svd28.h)


# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'during training ', 'after training'])
t.add_row(['training set', '%d %%' % acc_train_28[0],  round(max(acc_train_28), 2), '%d %%' % acc_train_28[-1]])
t.add_row(['validation set', '%d %%' % acc_val_28[0],  round(max(acc_val_28), 2), '%d %%' % acc_val_28[-1]])
print(t)


# k = 3
net_svd3 = SVDResNet(data_tsvd3, L)
torch.autograd.set_detect_anomaly(True)
_, acc_train_3, _, acc_val_3 = train(net_svd3,  max_epochs = epochs)
print(net_svd3.h)


# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'during training ', 'after training'])
t.add_row(['training set', '%d %%' % acc_train_3[0],  round(max(acc_train_3), 2), '%d %%' % acc_train_3[-1]])
t.add_row(['validation set', '%d %%' % acc_val_3[0],  round(max(acc_val_3), 2), '%d %%' % acc_val_3[-1]])
print(t)

"""

# Test how SVD on every time step performs ( SVD matrices  )

# k = 28
data_3svd28 = mnist(N, V, batch_size, k=28, transform = 'svd') ## data object
net_3svd28 = SVDResNet(data_3svd28, L)
torch.autograd.set_detect_anomaly(True)
_, acc_train3_28, _, acc_val3_28 = train(net_3svd28,  max_epochs = epochs)
print(net_3svd28.h)

# table for accuracy
print('\n statistics of accuracy')
t = PrettyTable(['', 'before training', 'during training ', 'after training'])
t.add_row(['training set', '%d %%' % acc_train3_28[0],  round(max(acc_train3_28),2), '%d %%' % acc_train3_28[-1]])
t.add_row(['validation set', '%d %%' % acc_val3_28[0],  round(max(acc_val3_28),2), '%d %%' % acc_val3_28[-1]])
print(t)


"""
#   NUMERICALLY UNSTABLE... :-S
#k = 3
data_svd3 = mnist(N, V, batch_size, k=3, transform = 'svd') ## data object
net_svd3 = SVDResNet(data_svd3, L)
print(" ----- TRAINING Net ------ ")
#torch.autograd.set_detect_anomaly(True)
start = time.time()
_, acc_train_3, _, acc_val_3 = train(net_svd3,  max_epochs = epochs)
end = time.time()
print("Training took ", round(end-start, 4), "seconds")
print(net_svd3.h)

"""

####### PLOTTING #########

fig = plt.figure()
# ORIG
"""
plt.plot(range(len(acc_train_orig)), acc_train_orig, 'tab:orange', label = 'orig')
plt.plot(range(len(acc_val_orig)), acc_val_orig, 'tab:orange', linestyle='--')
## IV SVD 28
plt.plot(range(len(acc_train_iv28)), acc_train_iv28, 'tab:purple', label = 'iv svd28' )
plt.plot(range(len(acc_val_iv28)), acc_val_iv28, 'tab:purple', linestyle='--')
##IV SVD 3
plt.plot(range(len(acc_train_iv3)), acc_train_iv3, 'tab:green', label = 'iv svd3 ')
plt.plot(range(len(acc_val_iv3)), acc_val_iv3, 'tab:green', linestyle='--')

# SVD 28 for all t
plt.plot(range(len(acc_train_28)), acc_train_28, 'tab:blue', label = 'tsvd 28 ')
plt.plot(range(len(acc_val_28)), acc_val_28, 'tab:blue', linestyle='--')
# SVD 3 for all t
plt.plot(range(len(acc_train_3)), acc_train_3, 'tab:red', label = 'tsvd 3 ')
plt.plot(range(len(acc_val_3)), acc_val_3, 'tab:red', linestyle='--')
# U, Sigma, V for all t
"""
plt.plot(range(len(acc_train3_28)), acc_train3_28, 'tab:gray', label = 'svd 28 ')
plt.plot(range(len(acc_val3_28)), acc_val3_28, 'tab:gray', linestyle='--')
# U, Sigma, V for all t (unstable)
# plt.plot(range(len(acc_train_3)), acc_train_3, 'tab:green', label = 'svd 3 ')
# plt.plot(range(len(acc_val_3)), acc_val_3, 'tab:green', linestyle='--')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title(r'MNIST  L = 10')
plt.savefig('MNIST_svd_investigation.png', bbox_inches='tight')
plt.show()



plt.title(r' $rank (X)$')
"""
plt.plot(net_orig.rank_evolution, 'tab:orange', label = 'orig')
plt.plot(net_iv28.rank_evolution, 'tab:purple', label = 'iv svd28')
plt.plot(net_iv3.rank_evolution, 'tab:green', label = 'iv svd3 ')
plt.plot(net_svd28.rank_evolution, 'tab:blue', label = 'tsvd 28 ')
plt.plot(net_svd3.rank_evolution, 'tab:red', label = 'tsvd 3 ')
"""
plt.plot(net_3svd28.rank_evolution, 'tab:gray', label = 'svd 28 ')
plt.legend()
plt.xlabel("layer of network")
plt.ylabel("average rank")
plt.show()

