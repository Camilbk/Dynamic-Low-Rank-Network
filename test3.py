
from data import svhn
from networks import DynTensorResNet
import optimisation
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure()
img = mpimg.imread('SVHN_AdversarialAttack_compressions.png')
imgplot = plt.imshow(img)
plt.show()