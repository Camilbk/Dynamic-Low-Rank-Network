# Dynamic-Low-Rank-Network

This repository contains code for testing the Dynamic Low Rank Networks. 
These networks are approximating the generated vector field created by residual neural networks, and training a 
classification function on a lower dimensional manifold. 

A demo can be found in Demo.ipynb and test.py. Demo investigates the rank of three datasets, MNIST FashionMNIST and
Grayscale Cifar10 and compares the approximation of the Dynamic low Rank Networks to the standard residual neural networks
for these three datasets. The file test.py trains a Dynamic Low Rank network on MNIST and shows some properties of the network,
such as orthogonality of the maatrices through the network, the rank evolution through the network, the structure of the networks,
and also plots the intergration error of the method. 

Good Luck! 