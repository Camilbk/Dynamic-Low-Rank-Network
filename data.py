import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Grayscale
from torch.utils.data import Dataset, DataLoader
from torch.linalg import svd, qr



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
        test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(test_dataloader))

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
        test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(test_dataloader))

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

class cifar10:
    """
    Prepares and transforms data to be ready for training

    https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition???

    """

    def __init__(self, N, V, batch_size, k=32, transform='none'):

        """
        Fetches data from CIFA and transformes
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

        # Training data
        training_data = datasets.CIFAR10(
            root="image_data",
            train=True,
            download=True,
            transform=Compose([ToTensor(), Grayscale(num_output_channels=1), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

        )

        # Validation data
        test_data = datasets.CIFAR10(
            root="image_data",
            train=False,
            download=True,
            transform=Compose([ToTensor(), Grayscale(num_output_channels=1), Lambda(lambda t: data_transform(transform, t, k))]),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )

        train_dataloader = DataLoader(training_data, batch_size=N, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=V, shuffle=True)

        X_train, C_train = next(iter(train_dataloader))
        X_val, C_val = next(iter(test_dataloader))

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
        s = "This is a CIFAR object: \n" \
            " size: (" + str(self.N) + "," + str(self.V) + ") \n" \
            " transform: " + self.transform + " with truncation k = " + str(self.k)+"\n" \
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
    if transform == 'none':
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
    p1d = (0, 0, 0, d - k)  ## Need to increase dimension
    sk = torch.nn.functional.pad(sk, p1d, "constant", 0)

    # For the network to be able to handle all three matrices through the network,
    #  we need them to be the same size when flattened.

    SVD[0] = torch.flatten(Uk[:, 0:k])
    SVD[1] = torch.flatten(sk)
    SVD[2] = torch.flatten(Vk[0:k, :])

    return SVD

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

