import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import Dataset, DataLoader
from torch.linalg import svd, qr
from torchvision.transforms.functional import rgb_to_grayscale

from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

class toydataset(Dataset):
    """Dataset for toydata.

    Parameters
    ----------
    X : torch.Tensor
        Data samples. Shape (num_samples, d).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
    """

    def __init__(self, X, C):
        self.X = X
        self.C = C

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'inputs': self.X[idx, :], 'labels': self.C[idx, :]}

        return sample


class mnist:
    """
    Prepares and transforms data to be ready for training

    """

    def __init__(self, N, V, batch_size, k='28', transform='none'):

        """
        Fetches data from MNIST and transformes
        :param N: samples for training
        :param V: samples for validation
        :param batch_size: batch size for sgd
        :param k: truncation for transform
        :param transform: data tranformation, ie. svd, polar decomp, qr, ...
        """

        self.N = N
        self.V = V
        self.batch_size = batch_size
        self.k = k
        self.n_channels = 1
        self.image_dimension = 28*28
        self.transform = transform
        self.labels_map = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }
        #self.all_data
        #self.TrainLoader
        #self.ValLoader
        k = 28 if transform == 'none' else k

        # Training data
        training_data = datasets.MNIST(
            root="image_data",
            train=True,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

        )

        # Validation data
        test_data = datasets.MNIST(
            root="image_data",
            train=False,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )

        train_dataloader = DataLoader(training_data, batch_size=N, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(self.test_dataloader))

        # create datasets
        TrainSet = toydataset(X_train, C_train)
        ValSet = toydataset(X_val, C_val)

        # create dataloaders for building batches
        self.TrainLoader = DataLoader(TrainSet, batch_size, shuffle=True)
        self.ValLoader = DataLoader(ValSet, batch_size, shuffle=True)

        self.all_data = [X_train, C_train, X_val, C_val]
        print("Data ready")

    @property
    def toString(self):
        s = "This is an MNIST object: \n" \
            " size: (" + str(self.N) + "," + str(self.V) + ") \n" \
            " transform: " + self.transform + " with truncation k = " + str(self.k)+"\n" \
            " batch size: " + str(self.batch_size)
        return s

class fashionMnist:
    """
    Prepares and transforms data to be ready for training

    """

    def __init__(self, N, V, batch_size, k='28', transform='none'):

        """
        Fetches data from MNIST and transformes
        :param N: samples for training
        :param V: samples for validation
        :param batch_size: batch size for sgd
        :param k: truncation for transform
        :param transform: data tranformation, ie. svd, polar decomp, qr, ...
        """

        self.N = N
        self.V = V
        self.batch_size = batch_size
        self.k = k
        self.n_channels = 1
        self.image_dimension = 28*28
        self.transform = transform
        self.labels_map = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }
        #self.all_data
        #self.TrainLoader
        #self.ValLoader
        k = 28 if transform == 'none' else k

        # Training data
        training_data = datasets.FashionMNIST(
            root="image_data",
            train=True,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

        )

        # Validation data
        test_data = datasets.FashionMNIST(
            root="image_data",
            train=False,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )

        train_dataloader = DataLoader(training_data, batch_size=N, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(self.test_dataloader))

        # create datasets (with possible transforms)
        TrainSet = toydataset(X_train, C_train)

        ValSet = toydataset(X_val, C_val)

        # create dataloaders for building batches
        self.TrainLoader = DataLoader(TrainSet, batch_size, shuffle=True)
        self.ValLoader = DataLoader(ValSet, batch_size, shuffle=True)

        self.all_data = [X_train, C_train, X_val, C_val]
        print("Data ready")

    @property
    def toString(self):
        s = "This is a Fashion MNIST object: \n" \
            " size: (" + str(self.N) + "," + str(self.V) + ") \n" \
            " transform: " + self.transform + " with truncation k = " + str(self.k)+"\n" \
            " batch size: " + str(self.batch_size)
        return s

class emnist:
    """

    Havent tested this

    Prepares and transforms data to be ready for training

    """

    def __init__(self, N, V, batch_size, k='28', transform='none'):

        """
        Fetches data from EMNIST and transformes
        :param N: samples for training
        :param V: samples for validation
        :param batch_size: batch size for sgd
        :param k: truncation for transform
        :param transform: data tranformation, ie. svd, polar decomp, qr, ...
        """

        self.N = N
        self.V = V
        self.batch_size = batch_size
        self.k = k
        self.n_channels = 1
        self.image_dimension = 28 * 28
        self.transform = transform
        self.labels_map = {
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "f",
            7: "g",
            8: "h",
            9: "i",
            10: "j",
            11: "k",
            12: "l",
            13: "m",
            14: "n",
            15: "o",
            16: "p",
            17: "q",
            18: "r",
            19: "s",
            20: "t",
            21: "u",
            22: "v",
            23: "w",
            24: "x",
            25: "y",
            26: "z"
        }

        k = 28 if transform == 'none' else k

        # Training data
        training_data = datasets.EMNIST(
            root="image_data",
            split = "letters",
            train = True,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                27, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

        )

        # Validation data
        test_data = datasets.EMNIST(
            root="image_data",
            split = "letters",
            train = False,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                27, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )

        train_dataloader = DataLoader(training_data, batch_size=N, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(self.test_dataloader))

        # create datasets (with possible transforms)
        TrainSet = toydataset(X_train, C_train)

        ValSet = toydataset(X_val, C_val)

        # create dataloaders for building batches
        self.TrainLoader = DataLoader(TrainSet, batch_size, shuffle=True)
        self.ValLoader = DataLoader(ValSet, batch_size, shuffle=True)

        self.all_data = [X_train, C_train, X_val, C_val]
        print("Data ready")

    @property
    def toString(self):
        s = "This is an EMNIST object: \n" \
            " size: (" + str(self.N) + "," + str(self.V) + ") \n" \
            " transform: " + self.transform + " with truncation k = " + str(self.k)+"\n" \
            " batch size: " + str(self.batch_size)
        return s

class cifar10:
    """
    Prepares and transforms data to be ready for training

    https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition???

    """

    def __init__(self, N, V, batch_size, k=32, transform='none'):

        """
        Fetches data from CIFAR and transformes
        :param N: samples for training
        :param V: samples for validation
        :param batch_size: batch size for sgd
        :param k: truncation for transform
        :param transform: data tranformation, ie. svd, polar decomp, qr, ...
        """

        self.N = N
        self.V = V
        self.batch_size = batch_size
        self.image_dimension = 32 * 32
        self.n_channels = 3
        self.k = k
        self.transform = transform
        self.labels_map = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        #self.all_data
        #self.TrainLoader
        #self.ValLoader
        k = 32 if transform == 'none' else k
        #assert (3 * k <= 32)  # for padding reason, the core is the problem here
        # Training data
        training_data = datasets.CIFAR10(
            root="image_data",
            train=True,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

        )

        # Validation data
        test_data = datasets.CIFAR10(
            root="image_data",
            train=False,
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )

        train_dataloader = DataLoader(training_data, batch_size=N, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(self.test_dataloader))

        # create datasets (with possible transforms)
        TrainSet = toydataset(X_train, C_train)

        ValSet = toydataset(X_val, C_val)

        # create dataloaders for building batches
        self.TrainLoader = DataLoader(TrainSet, batch_size, shuffle=True)
        self.ValLoader = DataLoader(ValSet, batch_size, shuffle=True)

        self.all_data = [X_train, C_train, X_val, C_val]
        print("Data ready")

    @property
    def toString(self):
        s = "This is a CIFAR10 object: \n" \
            " size: (" + str(self.N) + "," + str(self.V) + ") \n" \
            " transform: " + self.transform + " with truncation k = " + str(self.k)+"\n" \
            " batch size: " + str(self.batch_size)
        return s

class svhn:
    """
    Prepares and transforms data to be ready for training

    """

    def __init__(self, N, V, batch_size, k='32', transform='none'):
        """
        Fetches data from svhn and transformes
        :param N: samples for training
        :param V: samples for validation
        :param batch_size: batch size for sgd
        :param k: truncation for transform
        :param transform: data tranformation, ie. svd, polar decomp, qr, ...
        """

        self.N = N
        self.V = V
        self.batch_size = batch_size
        self.k = k
        self.image_dimension = 32 * 32
        self.n_channels = 3
        self.transform = transform
        self.labels_map = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }
        # self.all_data
        # self.TrainLoader
        # self.ValLoader
        k = 32 if transform == 'none' else k

        # Training data
        training_data = datasets.SVHN(
            root="image_data",
            split="train",
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

        )

        # Validation data
        test_data = datasets.SVHN(
            root="image_data",
            split="test",
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )

        train_dataloader = DataLoader(training_data, batch_size=N, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(self.test_dataloader))

        # create datasets (with possible transforms)
        TrainSet = toydataset(X_train, C_train)

        ValSet = toydataset(X_val, C_val)

        # create dataloaders for building batches
        self.TrainLoader = DataLoader(TrainSet, batch_size, shuffle=True)
        self.ValLoader = DataLoader(ValSet, batch_size, shuffle=True)

        self.all_data = [X_train, C_train, X_val, C_val]
        print("Data ready")

    @property
    def toString(self):
        s = "This is an SVHN object: \n" \
            " size: (" + str(self.N) + "," + str(self.V) + ") \n" \
                                                           " transform: " + self.transform + " with truncation k = " + str(
            self.k) + "\n" \
                      " batch size: " + str(self.batch_size)
        return s

def data_transform(transform, t, k):
    if transform == 'svd':
        return svd_transform(t, k)
    if transform == 'truncated svd':
        return truncated_svd(t,k)
    if transform == 'qr':
        return qr_decomp(t, k)
    if transform == 'polar':
        return polar_decomp(t, k)
    if transform == 'truncated tucker':
        return truncated_tucker(t, k)
    if transform == 'tucker':
        return tucker_decomposition(t,k)
    if transform == 'grayscale':
        return torch.flatten(rgb_to_grayscale(img = t, num_output_channels=1))
    if transform == 'none':
        if len(t) > 2:
            n_channels = t.shape[0]
            t = torch.reshape(t, (n_channels, k, k))
            return torch.flatten(t, 1,2)
        else:
            t = torch.reshape(t, (1,k,k))
            return torch.flatten(t, 1,2)

def svd_transform(t, k):
    """

    :param t: tensor (data/image)
    :param k: truncation
    :return: SVD
    """
    d = t.shape[1]
    # Store SVD of X in a data tensor
    SVD = torch.empty((3, d * k))
    ## For flat image in tensor

    t = torch.reshape(t, (d, d))  ## reshape
    Uk, sk, Vk = svd(t)  ## and find the truncates svd
    sk = torch.diag(sk[0:k])
    if k  != 28:
        p1d = (0, k - 3, 0, d - 3)  ## Need to increase dimension from (3,3) to (32, k)
        sk = torch.nn.functional.pad(sk, p1d, "constant", 0)

    # For the network to be able to handle all three matrices through the network,
    #  we need them to be the same size when flattened.

    SVD[0] = torch.flatten(Uk[:, 0:k])
    SVD[1] = torch.flatten(sk)
    SVD[2] = torch.flatten(Vk[0:k, :])

    return SVD

def tucker_decomposition(t, k):
    """

    :param t: tensor (data/image)
    :param k: truncation
    :return: SVD
    """
    # im = im.unflatten(-1,( 32,32))
    d = t.shape[1]  # d = 32
    assert (3 * k <= d)  # for padding reason, the core is the problem here

    ## Tucker decomposition is of the form [S, U_1, ..., U_N]
    # S is core and has same dim as ranks specified (3,k,k)
    # Us are factor matrices, and will be similar to truncated SVD size.
    core, factors = tucker(t.numpy(), rank=[3, k, k])  ## have to convert to numpy :-( .. workaround?
    U1 = factors[0]
    U2 = factors[1]
    U3 = factors[2]

    # Store tucker of X in a data tensor
    # S (3,k,k), U1 (3,3) U2 (32,k) U3 (32,k), d = 32    tucker_decomp:[S1,S2,S2, U1,U2,U3]
    # we can think of the core tensor to be "three" matrices stacked  of sizes k*k
    tucker_decomposition = torch.empty((6, d * k))
    ## For flat image in tensor
    S1 = core[0]
    S2 = core[1]
    S3 = core[2]
    ## all S matrices need to be padded an equal amount
    p1d = (0, 0, 0, d - k)  ## Need to increase dimension
    S1 = torch.nn.functional.pad(torch.from_numpy(S1), p1d, "constant", 0)  # adds rows of zeros
    S2 = torch.nn.functional.pad(torch.from_numpy(S2), p1d, "constant", 0)  # adds rows of zeros
    S3 = torch.nn.functional.pad(torch.from_numpy(S3), p1d, "constant", 0)  # adds rows of zeros
    p1d = (0, k - 3, 0, d - 3)  ## Need to increase dimension from (3,3) to (32, k)
    U1 = torch.nn.functional.pad(torch.from_numpy(U1), p1d, "constant", 0)

    # For the network to be able to handle all three matrices through the network,
    #  we need them to be the same size when flattened.
    tucker_decomposition[0] = torch.flatten(S1)
    tucker_decomposition[1] = torch.flatten(S2)
    tucker_decomposition[2] = torch.flatten(S3)
    tucker_decomposition[3] = torch.flatten(U1)  # this padding is potentially problematic, might break orthogonality property
    tucker_decomposition[4] = torch.flatten(torch.from_numpy(U2))
    tucker_decomposition[5] = torch.flatten(torch.from_numpy(U3))

    return tucker_decomposition

def from_tucker_decomposition(decomp, k):
    """
    converts from the padded version appropriate for pytorch to the tensorly version.
    used for recreating the images from the format used in pytorch. and also for remebering
    which matrices to truncate during the handling in the network.  :-)
    :param im:
    :return:
    """
    d = 32
    decomp = torch.reshape(decomp, (6, 32,k))
    S1 = decomp[0]
    S2 = decomp[1]
    S3 = decomp[2]
    S = torch.empty((3, d , k))
    S[0] = S1
    S[1] = S2
    S[2] = S3
    S = S[:,0:k, 0:k] # core, should be tucker rank
    U1 = decomp[3]
    U1 = U1[0:3,0:3]
    U2 = decomp[4]
    U3 = decomp[5]
    return S.detach().numpy(), [U1.detach().numpy(), U2.detach().numpy(), U3.detach().numpy()]

def truncated_svd(t,k):
    """
    :param t: torch tensor
    :param k: truncation k
    :return: one matrix of restores truncated svd tranformation
    """
    d = t.shape[1]
    u, s, vh = svd(t[0])
    SVD = torch.empty((1, d * d))
    SVD[0] = torch.flatten(u[:,0:k]@torch.diag(s[0:k])@vh[0:k,:])
    return SVD

def qr_decomp(t, k):
    """
    :param t: tensor (data/image)
    :param k: truncation
    :return: QR decomposition
    """
    d = t.shape[1]
    # Store SVD of X in a data tensor
    QR = torch.empty((2, d * d))
    ## For flat image in tensor
    t = torch.reshape(t, (d, d))  ## reshape
    Q, R = qr(t)

    QR[0] = torch.flatten(Q)  ## Orthogonal
    QR[1] = torch.flatten(R)  ## Upper Triangular

    return QR

def truncated_tucker(t,k):
    decomp = tucker(t.numpy(), rank=[3, k, k])
    t = tucker_to_tensor(decomp)
    return torch.flatten(torch.from_numpy(t).float(), 1,2)

def polar_decomp(t, k):
    """
    :param t: tensor (data/image)
    :param k: truncation
    :return: polar decomposition
    """
    d = t.shape[1]
    UP = torch.empty((2, d * d))
    ## For flat image in tensor
    t = torch.reshape(t, (d, d))  ## reshape
    ## X = U S V^T
    U, S, Vh = svd(t)

    ## P = VSV.T
    ## U = UV.T
    # X = UP

    P = Vh[0:k, :].T @ torch.diag(S[0:k]) @ Vh[0:k, :]
    U = U[:, 0:k] @ Vh[0:k, :]

    UP[0] = torch.flatten(U)
    UP[1] = torch.flatten(P)

    return UP

def restore_svd(data, k):
    u = data[0].reshape(28, k)
    s = data[1].reshape(28, k)
    s = s[0:k, 0:k]
    v = data[2].reshape(k, 28)

    return u @ s @ v