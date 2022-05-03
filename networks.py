import torch
import torch.nn as nn
from torch import eye, transpose, inverse
from torch.linalg import norm, svd, qr
import numpy as np
from matplotlib import pyplot as plt
from torch.linalg import matrix_rank
import prediction
from data import from_tucker_decomposition

import tensorly as tl
from tensorly import unfold
from tensorly.tenalg.core_tenalg import multi_mode_dot


def check_orthogonality(A):
    """
    :param A: matrix to be checked
    :return: ||I - A.T A||_F
    """
    return norm(eye(A.shape[1]) - A.T @ A)


def cay(C, D):
    """
    :param C: Matrix C (Tensor)
    :param D: Matic D (Tensor)
    :return: The cayley transformation of the two matrices

    Performs an alternative version of the Cayley transformation of B = CD.T
    """
    Id = torch.eye(C.shape[1])
    DT = torch.transpose(D, 1, 2)
    try:
        inv = torch.inverse(Id - 0.5 * DT @ C)
    except:
        print(Id - 0.5 * DT @ C)
    return Id + C @ inv @ DT


################################################################################
#
#        RESIDUAL NEURAL NET WITH TRAINABLE STEPSIZE
#
###############################################################################

class ResNet(nn.Module):
    """Euler network/Residual neural network.

    Parameters
    ----------
    data_object : data_object MNIST/CIFAR10
        Data object that comes with network
    L : int
        Number of layers.
    trainable_stepsize : boolean, optional
        If the stepsize in ResBet propagation should also be trained o not.
    d_hat : int, optional
        Dimension of feature space (here constant over all layers).

    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels
    and given by softmax function in case of general labels.
    """

    def __init__(self, data_object, L, trainable_stepsize=True, d_hat='none'):
        super(ResNet, self).__init__()

        self.d_hat = d_hat
        self.L = L
        self.k = data_object.k
        self.image_dimension = data_object.image_dimension
        self.transform = data_object.transform
        h = torch.Tensor(1)
        h[0] = 1 / L
        self.data_object = data_object
        self.h = nn.Parameter(h, requires_grad=True)  # random stepsize
        if not trainable_stepsize:
            self.h = 1  # Standard ResNet
        K = len(data_object.labels_map)
        batch_size, self.n_matrices, self.d = data_object.all_data[0].shape  # [1500, 3, 28*k] if  svd

        layers = [nn.Linear(self.image_dimension, self.image_dimension, bias=False)]  # input layer with no bias

        for layer in range(L):
            layers.append(nn.Linear(self.image_dimension, self.image_dimension))
        self.layers = nn.ModuleList(layers)
        # affine classifier
        self.classifier = nn.Linear(self.image_dimension, K)
        # activation function
        self.act = nn.ReLU()
        # hypothesis function to normalize prediction
        self.hyp = nn.Softmax(dim=1)

        self.layers[0].weight.requires_grad_(False)  # input layer not trained, i.e. weight is fixed

    def forward(self, X):
        """Propagate data through network.

        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).

        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output
            after applying classifier and hypothesis function). Shape
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape
            (num_samples, d_hat, L+1).
        """
        X_transformed = torch.empty(X.shape[0], self.image_dimension, self.L + 1)
        X = torch.reshape(X, (X.shape[0], self.image_dimension))
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            # input layer with no activation
            if i == 0:
                X = layer(X)
            # residual layers
            else:
                X = X + self.h * self.act(layer(X))
            X_transformed[:, :, i] = X

        # apply classifier
        X_classified = self.classifier(X)
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)

        return [X_predicted, X_classified, X_transformed]

    @property
    def rank_evolution(self):
        XsL = self.X_transformed[:, :, :]
        L = self.L
        ranks = np.zeros((XsL.shape[0], L + 1))
        for layer in range(L + 1):
            x = XsL[:, :, layer].unflatten(1, (self.d, self.d))
            ranks[:, layer] = matrix_rank(x)

        # print(err_U.shape)
        ranks = np.average(ranks, axis=0)
        plt.tight_layout()
        plt.title(r' $rank (X)$')
        plt.plot(ranks)
        plt.xlabel("layer of network")
        plt.ylabel("average rank")
        plt.show()
        return ranks

    @property
    def net_structure(self):
        print('network architecture:')
        print(self, '\n')

        # total number of neurons
        num_neurons = self.d + self.d * self.L
        print('total number of neurons in network:              ', num_neurons)

        # total number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total number of parameters in network:           ', num_params)
        print('total number of trainable parameters in network: ', num_trainable_params)

        s = 'This is a Residual neural network: \n' \
            ' depth: L = ' + str(self.L) + \
            ' activation function: ' + str(self.act) + \
            ' number of trainable parameters' + str(num_trainable_params)

        return s


################################################################################
#
#        RESIDUAL NEURAL NET WITH SVD IN EVERY LAYER
#
###############################################################################

class SVDResNet(nn.Module):
    """Euler network/Residual neural network.

    Parameters
    ----------
    data_object : data_object MNIST/CIFAR10
        Data object that comes with network
    L : int
        Number of layers.
    trainable_stepsize : boolean, optional
        If the stepsize in ResBet propagation should also be trained o not.
    d_hat : int, optional
        Dimension of feature space (here constant over all layers).

    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels
    and given by softmax function in case of general labels.
    """

    def __init__(self, data_object, L, trainable_stepsize=True, d_hat='none'):
        super(SVDResNet, self).__init__()
        self.d_hat = d_hat
        self.L = L
        self.k = data_object.k
        self.image_dimension = data_object.image_dimension  # typ 28*28 or 32*32
        self.dim = int(np.sqrt(self.image_dimension))
        self.transform = data_object.transform
        h = torch.Tensor(1)
        h[0] = 1 / L
        self.data_object = data_object
        self.h = nn.Parameter(h, requires_grad=True)  # random stepsize
        if not trainable_stepsize:
            self.h = 1  # Standard ResNet
        K = len(data_object.labels_map)
        batch_size, self.n_matrices, self.d = data_object.all_data[0].shape  # [1500, 3, 28*k] if  svd
        layers = []
        # input layer with no bias
        if self.transform == 'truncated svd':
            layers.append(nn.Linear(self.image_dimension, self.image_dimension, bias=False))
            for layer in range(L):
                layers.append(nn.Linear(self.image_dimension, self.image_dimension))
            self.layers = nn.ModuleList(layers)

        else:
            layers.append(nn.Linear(self.n_matrices * self.dim * self.k, self.n_matrices * self.dim * self.k, bias=False))
            for layer in range(L):
                layers.append(nn.Linear(self.n_matrices * self.dim * self.k, self.n_matrices* self.dim * self.k))
            self.layers = nn.ModuleList(layers)

        # affine classifier
        self.classifier = nn.Linear(self.image_dimension, K)
        # activation function
        self.act = nn.ReLU()
        # hypothesis function to normalize prediction
        self.hyp = nn.Softmax(dim=1)
        self.layers[0].weight.requires_grad_(False)  # input layer not trained, i.e. weight is fixed

    def forward(self, X):
        """Propagate data through network.

        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).

        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output
            after applying classifier and hypothesis function). Shape
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape
            (num_samples, d_hat, L+1).
        """

        def svd_projection(X):
            d = int(np.sqrt(self.image_dimension))
            u = X[:, 0, :].unflatten(1, (d, k))
            s = X[:, 1, :].unflatten(1, (d, k))
            vh = X[:, 2, :].unflatten(1, (k, d))
            image = (u[:, :, 0:k] @ s[:, 0:k, 0:k] @ vh[:, 0:k, :])
            u, s, vh = torch.svd(image)
            s = torch.diag_embed(s[:, 0:k])
            p1d = (0, 0, 0, d - k)  # Need to increase dimension
            s = torch.nn.functional.pad(s, p1d, "constant", 0)
            # For the network to be able to handle all three matrices through the network,
            #  we need them to be the same size when flattened.
            SVD = torch.empty((X.shape[0], 3, d * k))
            SVD[:, 0, :] = torch.flatten(u[:, :, 0:k], 1, 2)
            SVD[:, 1, :] = torch.flatten(s, 1, 2)
            SVD[:, 2, :] = torch.flatten(vh[:, 0:k, :], 1, 2)
            return SVD

        k = self.k
        if self.n_matrices == 1:
            X = torch.reshape(X, (X.shape[0], self.image_dimension))
            X_transformed = torch.empty(X.shape[0], self.image_dimension, self.L + 1)
            # propagate data through network (forward pass)
            for i, layer in enumerate(self.layers):
                # input layer with no activation
                d = int(np.sqrt(self.image_dimension))
                if i == 0:
                    X = layer(X)
                    u, s, vh = svd(X.unflatten(1, (d, d)))
                    X = torch.flatten(u[:, :, 0:k] @ torch.diag_embed(s[:, 0:k]) @ vh[:, 0:k, :], 1, 2)
                # residual layers
                else:
                    X = X + self.h * self.act(layer(X))
                    u, s, vh = svd(X.unflatten(1, (d, d)))
                    X = torch.flatten(u[:, :, 0:k] @ torch.diag_embed(s[:, 0:k]) @ vh[:, 0:k, :], 1, 2)
                X_transformed[:, :, i] = X
        else:
            X_transformed = torch.empty(X.shape[0], self.n_matrices, self.dim * self.k, self.L + 1)
            # propagate data through network (forward pass)
            for i, layer in enumerate(self.layers):
                print('X', X.shape)
                # input layer with no activation
                if i == 0:
                    X = layer(X)
                    # X = svd_projection(X)
                # residual layers
                else:
                    X = X + self.h * self.act(layer(X))
                    X = svd_projection(X)
                X_transformed[:, :, :, i] = X

        if self.n_matrices != 1:
            u = X[:, 0, :].unflatten(1, (self.dim, k))
            s = X[:, 1, :].unflatten(1, (self.dim, k))
            vh = X[:, 2, :].unflatten(1, (k, self.dim))
            X = torch.flatten(u @ s[:, 0:k, 0:k] @ vh, 1, 2)

        # apply classifier
        X_classified = self.classifier(X)
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)
        return [X_predicted, X_classified, X_transformed]

    @property
    def net_orthogonality(self):
        # Investigating orthogonality of matrices through the network
        if self.transform == 'svd':  # [1500, 3, 28*k, 101]
            UsL = self.X_transformed[:, 0, :, :]
            VsL = self.X_transformed[:, 2, :, :]
            # print("UsL", UsL.shape)
            L = self.L
            k = self.k
            err_U = np.zeros((UsL.shape[0], L + 1))
            err_V = np.zeros((VsL.shape[0], L + 1))
            for i, u_evol in enumerate(UsL):
                v_evol = VsL[i]  # pick a matrix
                for layer in range(L + 1):
                    u = u_evol[:, layer]
                    v = v_evol[:, layer]
                    err_U[i, layer] = check_orthogonality(u.unflatten(0, (self.dim, k)))
                    err_V[i, layer] = check_orthogonality(v.unflatten(0, (self.dim, k)))
            # print(err_U.shape)
            err_U = np.average(err_U, axis=0)
            err_V = np.average(err_V, axis=0)
            # print(err.shape)

            fig, ax = plt.subplots(2)
            plt.tight_layout()

            ax[0].set_title(r' $|| I - U^T U ||_F$')
            ax[0].plot(err_U)
            ax[0].set_xlabel("layer of network")
            ax[0].set_ylabel("error")

            ax[1].set_title(r' $|| I - V^T V ||_F$')
            ax[1].plot(err_V)
            ax[1].set_xlabel("layer of network")
            ax[1].set_ylabel("error")
            plt.show()

            s = ' highest average orthogonality error in U: ' + str(max(err_U)) + \
                '\n highest average orthogonality error in V: ' + str(max(err_V))

            return s, err_U, err_V

    @property
    def net_structure(self):
        print('network architecture:')
        print(self, '\n')

        # total number of neurons
        num_neurons = self.d + self.d * self.L
        print('total number of neurons in network: ', num_neurons)

        # total number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total number of parameters in network:           ', num_params)
        print('total number of trainable parameters in network: ', num_trainable_params)

        s = 'This is a Residual neural network: \n' \
            ' depth: L = ' + str(self.L) + \
            ' activation function: ' + str(self.act) + \
            ' number of trainable parameters' + str(num_trainable_params)

        return s

    @property
    def net_rank_evolution(self):
        if self.transform == 'truncated svd':
            XsL = self.X_transformed[:, :, :]
            L = self.L
            ranks = np.zeros((XsL.shape[0], L + 1))
            for layer in range(L + 1):
                x = XsL[:, :, layer].unflatten(1, (self.d, self.d))
                ranks[:, layer] = matrix_rank(x)

            # print(err_U.shape)
            ranks = np.average(ranks, axis=0)
            plt.tight_layout()
            plt.title(r' $rank (X)$')
            plt.plot(ranks)
            plt.xlabel("layer of network")
            plt.ylabel("average rank")
            plt.show()
            return ranks

        elif self.transform == 'svd':
            UsL = self.X_transformed[:, 0, :, :]
            SsL = self.X_transformed[:, 1, :, :]
            VsL = self.X_transformed[:, 2, :, :]

            L = self.L
            k = self.k
            ranks = np.zeros((UsL.shape[0], L + 1))
            for layer in range(L + 1):
                u = UsL[:, :, layer].unflatten(1, (self.dim, k))
                s = SsL[:, :, layer].unflatten(1, (self.dim, k))
                s = s[:, 0:k, 0:k]
                v = VsL[:, :, layer].unflatten(1, (self.dim, k))
                x = u @ s @ torch.transpose(v, 1, 2)
                ranks[:, layer] = matrix_rank(x)
            return ranks

    @property
    def net_validate(self):
        X_val = self.data_object.all_data[2]
        ValLoader = self.data_object.ValLoader
        compression_type = self.data_object.transform
        X_predicted, _, _ = self(X_val)

        # visualization of some example images
        batch_num = 3
        ValIter = iter(ValLoader)
        example_batches = []
        cols, rows = self.batch_size, batch_num
        figure = plt.figure(figsize=(cols * 2, rows * 3))

        if compression_type == "none" or compression_type == "truncated svd":
            for j in range(batch_num):
                sample_batched = ValIter.next()
                example_batches.append(sample_batched)
                X_predicted_batch, _, _ = self(sample_batched['inputs'])
                C_pred_batch = prediction.pred(X_predicted_batch)
                for i in range(self.batch_size):
                    figure.add_subplot(rows, cols, j * self.batch_size + i + 1)
                    image = sample_batched['inputs'][i]
                    true_label = sample_batched['labels'][i]
                    predicted_label = C_pred_batch[i]
                    true_label_name = self.labels_map[true_label.nonzero(as_tuple=False)[0][0].item()]
                    predicted_label_name = self.labels_map[predicted_label.nonzero(as_tuple=False)[0][0].item()]
                    plt.title('true: %s \n predicted: %s' % (true_label_name, predicted_label_name))
                    plt.axis("off")
                    plt.imshow(image.view(28, 28).squeeze(), cmap="gray")
            plt.show()

        # do sanity check
        print('sanity check - prediction mean should be roughly %.4f and prediction variance relatively small' % (
            1 / 2 if self.K == 1 else 1 / self.K))
        print('prediction mean:     ', X_predicted.mean(axis=0).data)
        print('prediction variance: ', X_predicted.std(axis=0).data)
        return ':-)'


################################################################################
#
#            DYNAMIC LOW RANK APPROXIMATION NETWORK
#
###############################################################################

class DynResNet(nn.Module):
    """Residual neural network for dynamic low rank approximation of input.

    Parameters
    ----------

    data_object : data_object MNIST/CIFAR10
        Data object that comes with network
    L : int
        Number of layers.
    d_hat : int, optional
        Dimension of feature space (here constant over all layers).

    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels
    and given by softmax function in case of general labels.
    """

    def __init__(self, data_object, L, d_hat='none'):
        super(DynResNet, self).__init__()

        assert data_object.transform == 'svd'

        self.data_object = data_object
        self.k = data_object.k
        self.d_hat = d_hat
        self.L = L
        # h = torch.rand(1)
        # h = torch.Tensor(1)
        # h[0] = 1/L
        self.h = 0.001  # [0.005, 0.03]
        # self.h = nn.Parameter(h, requires_grad = True) # random stepsize
        K = len(data_object.labels_map)
        batch_size, n_matrices, self.d = data_object.all_data[0].shape  # [1500, 3, 28*k] if  svd
        if self.d_hat == 'none':
            self.d_hat = self.d

        assert self.d_hat >= self.d
        self.dim = int(self.d_hat / self.k)
        # L layers defined by affine operation z = Ky + b
        layers = [nn.Linear(self.dim * self.dim, self.dim * self.dim, bias=False)]  # input layer with no bias

        for layer in range(L):
            layers.append(nn.Linear(self.dim * self.dim, self.dim * self.dim))
        self.layers = nn.ModuleList(layers)

        # affine classifier
        self.classifier = nn.Linear(self.dim * self.dim, K)
        # activation function
        self.act = nn.ReLU()
        # hypothesis function to normalize prediction
        self.hyp = nn.Softmax(dim=1)

        self.layers[0].weight.requires_grad_(False)  # input layer not trained, i.e. weight is fixed

        print("Network ready")

    def forward(self, X):
        """Propagate data through network.

        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).

        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output
            after applying classifier and hypothesis function). Shape
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape
            (num_samples, d_hat, L+1).
        """
        def invert_S(s):
            s_inv = torch.empty(s.shape)
            for i, m in enumerate(s):
                diag = torch.diagonal(m)
                diag_inv = diag ** (-1)
                s_inv[i] = torch.diag(diag_inv)
            return s_inv

        k = self.k

        X_transformed = torch.zeros(X.shape[0], self.dim * self.dim, self.L + 1)
        self.Us = torch.zeros(X.shape[0], self.dim * k, self.L + 1)  # U(t)
        Ss = torch.zeros(X.shape[0], k * k, self.L + 1)  # S(t)
        self.Vs = torch.zeros(X.shape[0], self.dim * k, self.L + 1)  # V(t)

        # u, s, vh = svd(X) # inital guess (truncated)
        u = X[:, 0].reshape((X.shape[0], self.dim, k))
        s = X[:, 1].reshape(X.shape[0], self.dim, k)
        s = s[:, 0:k, 0:k]
        vh = X[:, 2].reshape(X.shape[0], k, self.dim)
        v = torch.transpose(vh, 1, 2)

        Ss[:, :, 0] = torch.flatten(s, 1, 2)
        self.Us[:, :, 0] = torch.flatten(u, 1, 2)
        self.Vs[:, :, 0] = torch.flatten(v, 1, 2)

        X = u @ s @ vh
        X = torch.flatten(X, 1, 2)

        # Checking of the derivatives lie in the tangent space
        self.invariantsU = torch.zeros(X.shape[0], self.L + 1)
        self.invariantsV = torch.zeros(X.shape[0], self.L + 1)

        self.integration_error = torch.empty(X.shape[0], 2, self.L + 1)

        # print(self.h)
        # lagrange_multipliers = True
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            # time integration and projection on stiefel
            dY = self.act(layer(X))
            dY = dY.unflatten(1, (self.dim, self.dim))
            s_inv = invert_S(s)
            dU = dY @ v @ s_inv
            dV = transpose(dY, 1, 2) @ u @ transpose(s_inv, 1, 2)
            dS = transpose(u, 1, 2) @ dY @ v

            # cay( F U.T - U F.T ) U

            # Projection
            Id = torch.eye(u.shape[1])
            Id = Id.reshape((1, u.shape[1], u.shape[1]))
            Id = Id.repeat(u.shape[0], 1, 1)
            uT = torch.transpose(u, 1, 2)
            F = (Id - u @ uT) @ dU
            FT = torch.transpose(F, 1, 2)
            u_tilde = self.h * (F @ uT - u @ FT) @ u  # Propagation in direction of tangent
            u = cay(self.h * (F @ uT), - self.h * (u @ FT)) @ u  # Project onto stiefel from tangent space
            self.integration_error[:, 0, i] = norm(u - u_tilde)
            ######
            Id = torch.eye(v.shape[1])  # kan lagre I i minne så man slipper å lage denne hver gang
            Id = Id.reshape((1, v.shape[1], v.shape[1]))
            Id = Id.repeat(v.shape[0], 1, 1)
            vT = torch.transpose(v, 1, 2)
            F = (Id - v @ vT) @ dV
            FT = torch.transpose(F, 1, 2)
            v_tilde = self.h * (F @ vT - v @ FT) @ v  # Propagation in direction of tangent
            v = cay(self.h * (F @ vT), - self.h * (v @ FT)) @ v  # Project onto stiefel from tangent space
            self.integration_error[:, 1, i] = norm(v - v_tilde)
            # u = tangent_projection(u, dU, self.h)
            # v = tangent_projection(v, dV, self.h)
            ######

            s = s + self.h * dS

            X = u @ s @ transpose(v, 1, 2)

            # Invariants in tangent space if dY^T Y + Y^ dY
            self.invariantsU[:, i] = norm(torch.transpose(dU, 1, 2) @ u) - norm(dU @ torch.transpose(u, 1, 2))
            self.invariantsV[:, i] = norm(torch.transpose(dV, 1, 2) @ v) - norm(dV @ torch.transpose(v, 1, 2))

            X = torch.flatten(X, 1, 2)

            Ss[:, :, i] = torch.flatten(s, 1, 2)
            self.Us[:, :, i] = torch.flatten(u, 1, 2)
            self.Vs[:, :, i] = torch.flatten(v, 1, 2)

            X_transformed[:, :, i] = X

        # apply classifier
        X_classified = self.classifier(X)  # Projects dimension of network down to output space. Linear classifier
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)  # Maps a batch into a probability distribution

        return [X_predicted, X_classified, X_transformed]

    @property
    def rank_evolution(self):
        XsL = self.X_transformed[:, :, :]
        L = self.L
        ranks = np.zeros((XsL.shape[0], L + 1))
        for layer in range(L + 1):
            x = XsL[:, :, layer].unflatten(1, (self.d, self.d))
            ranks[:, layer] = matrix_rank(x)

        # print(err_U.shape)
        ranks = np.average(ranks, axis=0)
        plt.tight_layout()
        plt.title(r' $rank (X)$')
        plt.plot(ranks)
        plt.xlabel("layer of network")
        plt.ylabel("average rank")
        plt.show()
        return ranks

    @property
    def orthogonality(self):
        classname = self.data_object.__class__.__name__
        # Investigating orthogonality of matrices through the network
        UsL = self.Us  # [1500, 28*k, L +1 ]
        VsL = self.Vs
        L = self.L
        k = self.k
        err_U = np.zeros((UsL.shape[0], L + 1))
        err_V = np.zeros((VsL.shape[0], L + 1))
        for i, u_evol in enumerate(UsL):
            # matrix_U = u_evol  # pick a matrix
            v_evol = VsL[i]  # pick a matrix
            for layer in range(L + 1):
                u = u_evol[:, layer]
                v = v_evol[:, layer]
                err_U[i, layer] = check_orthogonality(u.unflatten(0, (self.dim, k)))
                err_V[i, layer] = check_orthogonality(v.unflatten(0, (self.dim, k)))

        err_U = np.average(err_U, axis=0)
        err_V = np.average(err_V, axis=0)
        # print(err.shape)
        # err_av = err/len(Us)
        # plt.title(r'Orthogonality evolution $|| U U^T - I||_F$')

        fig, ax = plt.subplots(2)
        plt.tight_layout()

        ax[0].set_title(r' $|| I - U^T U ||_F$')
        ax[0].plot(err_U)
        ax[0].set_xlabel("layer of network")
        ax[0].set_ylabel("error")

        ax[1].set_title(r' $|| I - V^T V ||_F$')
        ax[1].plot(err_V)
        ax[1].set_xlabel("layer of network")
        ax[1].set_ylabel("error")
        plt.savefig('DynLowRank_Orth%i_%s.png' % (L, classname), bbox_inches='tight')

        s = ' highest average orthogonality error in U: ' + str(max(err_U)) + \
            '\n highest average orthogonality error in V: ' + str(max(err_V))

        return s, err_U, err_V

    @property
    def net_structure(self):
        print('network architecture:')
        print(self, '\n')

        # total number of neurons
        num_neurons = self.d + self.d_hat * self.L
        print('total number of neurons in network:              ', num_neurons)

        # total number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total number of parameters in network:           ', num_params)
        print('total number of trainable parameters in network: ', num_trainable_params)

        s = 'This is a Dynamic low rank network: \n' \
            ' depth: L = ' + str(self.L) + '\n' \
            ' activation function: ' + str(self.act) + '\n' \
            ' number of trainable parameters: ' + str(num_trainable_params)
        return s

    @property
    def plot_integration_error(self):
        # want to track how bad the numerical integration projects away from the Stiefel manifold
        # Could maybe be used to reject timesteps, and try with a smaller step h, but still we
        # have very limited control of the error in each step

        err_U = np.average(self.integration_error[:, 0, :], axis=0)
        err_V = np.average(self.integration_error[:, 1, :], axis=0)
        plt.tight_layout()
        plt.title(r' Integration error')
        plt.plot(err_U, label=r'$|| U - \tilde U ||_F$')
        plt.plot(err_V, label=r'$|| V - \tilde V ||_F$')
        plt.xlabel("layer of network")
        plt.ylabel(" average error")
        plt.show()

        return err_U, err_V


################################################################################
#
#            DYNAMIC TENSOR APPROXIMATION NETWORK
#
###############################################################################

class DynTensorResNet(nn.Module):
    """Residual neural network for dynamic low rank approximation of tensor input.

    Parameters
    ----------

    data_object : data_object CIFAR10
        Data object that comes with network
    L : int
        Number of layers.
    d_hat : int, optional
        Dimension of feature space (here constant over all layers).

    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels
    and given by softmax function in case of general labels.
    """

    def __init__(self, data_object, L, h = 0.001, d_hat='none'):
        super(DynTensorResNet, self).__init__()

        #assert data_object.transform == 'tucker'
        self.n_channels = 3
        self.h = 1/L
        self.data_object = data_object
        self.k = data_object.k
        self.d_hat = d_hat
        self.L = L
        # h = torch.rand(1)
        # h = torch.Tensor(1)
        # h[0] = 1/L
        #self.h = 0.001  # [0.005, 0.03]
        #self.h = nn.Parameter(self.h, requires_grad = True) # random stepsize
        K = len(data_object.labels_map)
        batch_size, n_matrices, self.d = data_object.all_data[0].shape  # [1500, 3, 28*k] if  svd
        if self.d_hat == 'none':
            self.d_hat = self.d

        assert self.d_hat >= self.d
        self.dim = int(self.d_hat / self.k)
        # L layers defined by affine operation z = Ky + b
        in_features =  self.n_channels*self.dim * self.dim
        out_features =  self.n_channels*self.dim * self.dim
        layers = [nn.Linear(in_features= in_features, out_features=out_features, bias=False)]  # input layer with no bias

        for layer in range(L):
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

        # affine classifier
        self.classifier = nn.Linear(self.n_channels* self.dim * self.dim, K)
        # activation function
        self.act = nn.ReLU()
        # hypothesis function to normalize prediction
        self.hyp = nn.Softmax(dim=1)

        self.layers[0].weight.requires_grad_(False)  # input layer not trained, i.e. weight is fixed

        print("Network ready")

    def forward(self, X):
        """Propagate data through network.

        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).

        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output
            after applying classifier and hypothesis function). Shape
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape
            (num_samples, d_hat, L+1).
        """

        k = self.k

        self.X_transformed = torch.zeros(X.shape[0], self.n_channels*  self.dim * self.dim, self.L + 1)  #[1500, 3, 32*32, layers]
        Ss = torch.zeros(X.shape[0], self.n_channels, self.k*self.k , self.L + 1)  # S(t) core tensor [1500, 3, (k, k), layers]
        self.U1s = torch.zeros(X.shape[0], self.n_channels * self.n_channels, self.L + 1)  # U1(t)  [1500, (3, 3), layers ]
        self.U2s = torch.zeros(X.shape[0], self.dim * k, self.L + 1)  # U2(t)  [1500, (32, k), layers ]
        self.U3s = torch.zeros(X.shape[0], self.dim * k, self.L + 1)  # U3(t) [1500, (32, k), layers ]

        #S1, S2, S3, (core)  U1, U2, U3  (factors)   inital guess (truncated)
        def restore_core(X, d, k ):
            S1 = X[:, 0]
            S2 = X[:, 1]
            S3 = X[:, 2]
            S = torch.empty((X.shape[0], 3, d, k))
            S[:, 0] = S1
            S[:, 1] = S2
            S[:, 2] = S3
            return S[:, :, 0:k, 0:k]  # core, should be tucker rank

        def prep_for_vec_field(X):
            restored_im = torch.empty((X.shape[0], self.n_channels*self.dim*self.dim))
            for i, decomp in enumerate(X): # loop over batch :-(
                decomp = from_tucker_decomposition(decomp, self.k)
                tensor = tl.tucker_to_tensor(decomp)
                restored_im[i] = torch.flatten(torch.from_numpy(tensor), 0, -1)
            return restored_im

        def Sn_dagger(Sn):
            # Sn : [1500, dim, dim]
            # takes in an n'th unfolding of the S matrix and creates the pseudoinv
            SnT = Sn.T
            result = SnT @ np.linalg.inv(Sn @ SnT)
            if len(np.argwhere(np.isnan(result))) == 0:
                return result
            else: print("Nan detected in inverse", np.argwhere(np.isnan(result)))

        def Id(U):
            Id = torch.eye(U.shape[1])
            Id = Id.reshape((1, U.shape[1], U.shape[1]))
            Id = Id.repeat(U.shape[0], 1, 1)
            return Id

        X = X.unflatten(-1, (self.dim, self.k)) # [batch, 6, (32, k)]
        S = restore_core(X, self.dim, self.k)
        U1 = X[:, 3]
        U1 = U1[:, 0:self.n_channels, 0:self.n_channels]
        U2 = X[:, 4]
        U3 = X[:, 5]

        Ss[:, :, :,  0] = torch.flatten(S, 2,3)
        self.U1s[:, :, 0] = torch.flatten(U1, 1, 2)
        self.U2s[:, :, 0] = torch.flatten(U2, 1, 2)
        self.U3s[:, :, 0] = torch.flatten(U3, 1, 2)

        self.integration_error = torch.empty(X.shape[0], 3, self.L + 1)

        dS = np.zeros(S.shape)
        dU1 = np.zeros(U1.shape)
        dU2 = np.zeros(U2.shape)
        dU3 = np.zeros(U3.shape)

        X = prep_for_vec_field(X) # flatted restored "truncated" tucker decomposition
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            if i == 0:
                X = layer(X)
            else:
                # time integration and projection on stiefel
                dY = self.act(layer( X ))  #[1500, 3 * 32* 32]
                dY = dY.unflatten(-1, (self.n_channels, self.dim,self.dim)) #[1500, 3, 32, 32]

                ### each U_n is multiplied by the nth-unfolding of S dagger
                # will not try to be smartypants here so will do it so to make sure the implementation is correct

                U1T = transpose(U1,1,2)
                U2T = transpose(U2, 1, 2)
                U3T = transpose(U3, 1, 2)

                dY_numpy = dY.detach().clone().numpy()
                S_numpy = S.numpy()
                U1T_numpy = U1T.numpy()
                U2T_numpy = U2T.numpy()
                U3T_numpy = U3T.numpy()
                for el in range(X.shape[0]):  # for element in batch :-(
                    dS[el] = multi_mode_dot(dY_numpy[el], [ U1T_numpy[el], U2T_numpy[el], U3T_numpy[el] ])
                    dU1[el] = unfold(multi_mode_dot(dY_numpy[el], [ U2T_numpy[el], U3T_numpy[el]], modes=[1, 2]), 0) @ Sn_dagger( unfold(S_numpy[el], 0))
                    dU2[el] = unfold( multi_mode_dot(dY_numpy[el], [ U1T_numpy[el], U3T_numpy[el]] , modes=[0, 2]), 1) @ Sn_dagger( unfold(S_numpy[el], 1))
                    dU3[el] = unfold( multi_mode_dot(dY_numpy[el], [ U1T_numpy[el], U2T_numpy[el]] , modes=[0, 1]), 2) @ Sn_dagger( unfold(S_numpy[el], 2))

                # cay( F U.T - U F.T ) U
                # Projection
                IdU23 = Id(U2)
                FU1 = ( Id(U1) - U1 @ U1T) @ torch.from_numpy(dU1).float()
                FU2 = (IdU23 - U2 @ U2T) @ torch.from_numpy(dU2).float()
                FU3 = (IdU23 - U3 @ U3T) @ torch.from_numpy(dU3).float()

                FU1T = torch.transpose(FU1, 1, 2)
                FU2T = torch.transpose(FU2, 1, 2)
                FU3T = torch.transpose(FU3, 1, 2)

                U1 = cay(FU1 @ U1T, - self.h * (U1 @ FU1T)) @ U1  # Project onto stiefel from tangent space
                print(check_orthogonality(U1))
                U2 = cay(self.h * (FU2 @ U2T), - self.h * (U2 @ FU2T)) @ U2  # Project onto stiefel from tangent space
                U3 = cay(self.h * (FU3 @ U3T), - self.h * (U3 @ FU3T)) @ U3  # Project onto stiefel from tangent space

                #self.integration_error[:, 0, i] = norm(u - u_tilde)

                S = S + self.h * dS

                # update the image
                restored_im = torch.empty((X.shape[0], self.n_channels*self.dim*self.dim))
                for im in range(X.shape[0]): #loop over batch
                    tensor = tl.tucker_to_tensor((S[im].numpy(), [U1[im].numpy(), U2[im].numpy(), U3[im].numpy()]))
                    restored_im[im] = torch.flatten(torch.from_numpy(tensor), 0, -1)

                X = restored_im

                # Invariants in tangent space if dY^T Y + Y^ dY
                #self.invariantsU[:, i] = norm(torch.transpose(dU, 1, 2) @ u) - norm(dU @ torch.transpose(u, 1, 2))
                #self.invariantsV[:, i] = norm(torch.transpose(dV, 1, 2) @ v) - norm(dV @ torch.transpose(v, 1, 2))
                Ss[:, :, :, i ] = torch.flatten(S, 2, 3)
                self.U1s[:, :, i ] = torch.flatten(U1, 1, 2)
                self.U2s[:, :, i] = torch.flatten(U2, 1, 2)
                self.U3s[:, :, i ] = torch.flatten(U3, 1, 2)

            self.X_transformed[:, :, i ] = X

        # apply classifier
        X_classified = self.classifier(X)  # Projects dimension of network down to output space. Linear classifier
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)  # Maps a batch into a probability distribution

        return [X_predicted, X_classified, self.X_transformed]

    @property
    def rank_evolution(self):
        XsL = self.X_transformed[:, :, :]
        L = self.L
        ranks = np.zeros((XsL.shape[0], L + 1))
        for layer in range(L + 1):
            x = XsL[:, :, layer].unflatten(1, (self.d, self.d))
            ranks[:, layer] = matrix_rank(x)

        # print(err_U.shape)
        ranks = np.average(ranks, axis=0)
        plt.tight_layout()
        plt.title(r' $rank (X)$')
        plt.plot(ranks)
        plt.xlabel("layer of network")
        plt.ylabel("average rank")
        plt.show()
        return ranks

    @property
    def orthogonality(self):
        classname = self.data_object.__class__.__name__
        # Investigating orthogonality of matrices through the network
        U1sL = self.U1s  # U1(t)  [1500,  (3,3) , layers  ]
        U2sL = self.U2s  # U2(t)  [1500, (32, k), layers ]
        U3sL = self.U3s  # U3(t)  [1500, (32, k), layers ]
        L = self.L
        k = self.k
        err_U1 = np.zeros((U1sL.shape[0], L ))
        err_U2 = np.zeros((U2sL.shape[0], L  ))
        err_U3 = np.zeros((U3sL.shape[0], L ))
        for i, u1_evol in enumerate(U1sL):
            # matrix_U = u_evol  # pick a matrix
            u2_evol = U2sL[i]  # pick a matrix
            u3_evol = U3sL[i]  # pick a matrix

            for layer in range(L ):
                u1 = u1_evol[:, layer]
                u2 = u2_evol[:, layer]
                u3 = u3_evol[:, layer]

                err_U1[i, layer] = check_orthogonality(u1.unflatten(0, (self.n_channels, self.n_channels)))
                err_U2[i, layer] = check_orthogonality(u2.unflatten(0, (self.dim, k)))
                err_U3[i, layer] = check_orthogonality(u3.unflatten(0, (self.dim, k)))

        #print( "1",err_U1, "\n 2" , err_U2,"\n 3 ",  err_U3)

        err_U1 = np.average(err_U1, axis=0)
        err_U2 = np.average(err_U2, axis=0)
        err_U3 = np.average(err_U3, axis=0)
        # print(err.shape)
        # err_av = err/len(Us)
        # plt.title(r'Orthogonality evolution $|| U U^T - I||_F$')

        fig, ax = plt.subplots(3)
        plt.tight_layout()

        ax[0].set_title(r' $|| I - U_1^T U_1 ||_F$')
        ax[0].plot(err_U1)
        ax[0].set_xlabel("layer of network")
        ax[0].set_ylabel("error")

        ax[1].set_title(r' $|| I - U2^T U2 ||_F$')
        ax[1].plot(err_U2)
        ax[1].set_xlabel("layer of network")
        ax[1].set_ylabel("error")

        ax[2].set_title(r' $|| I - U_3^T U_3 ||_F$')
        ax[2].plot(err_U2)
        ax[2].set_xlabel("layer of network")
        ax[2].set_ylabel("error")
        plt.show()
        plt.savefig('DynTensorRank_Orth%i_%s.png' % (L, classname), bbox_inches='tight')


        s = ' highest average orthogonality error in U1: ' + str(max(err_U1)) + \
            '\n highest average orthogonality error in U2: ' + str(max(err_U2)) + \
            '\n highest average orthogonality error in U3: ' + str(max(err_U3))

        return s, err_U1, err_U2, err_U3

    @property
    def net_structure(self):
        print('network architecture:')
        print(self, '\n')

        # total number of neurons
        num_neurons = self.d + self.d_hat * self.L
        print('total number of neurons in network:              ', num_neurons)

        # total number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total number of parameters in network:           ', num_params)
        print('total number of trainable parameters in network: ', num_trainable_params)

        s = 'This is a Dynamic low rank network: \n' \
            ' depth: L = ' + str(self.L) + '\n' \
            ' activation function: ' + str(self.act) + '\n' \
            ' number of trainable parameters: ' + str(num_trainable_params)
        return s

    @property
    def plot_integration_error(self):
        # want to track how bad the numerical integration projects away from the Stiefel manifold
        # Could maybe be used to reject timesteps, and try with a smaller step h, but still we
        # have very limited control of the error in each step

        err_U = np.average(self.integration_error[:, 0, :], axis=0)
        err_V = np.average(self.integration_error[:, 1, :], axis=0)
        plt.tight_layout()
        plt.title(r' Integration error')
        plt.plot(err_U, label=r'$|| U - \tilde U ||_F$')
        plt.plot(err_V, label=r'$|| V - \tilde V ||_F$')
        plt.xlabel("layer of network")
        plt.ylabel(" average error")
        plt.show()

        return err_U, err_V



################################################################################
#
#           PROJECTION NETWORK
#
###############################################################################

class ProjResNet(nn.Module):
    """Euler network/Residual neural network.

    Parameters
    ----------
    data_object : data_object MNIST/CIFAR10
        Data object that comes with network
    L : int
        Number of layers.
    d_hat : int, optional
        Dimension of feature space (here constant over all layers).
    projection_type : string, optional
        projection type that will be performed at every timestep, default is polar
    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels
    and given by softmax function in case of general labels.
    """

    def __init__(self, data_object, L, d_hat='none', projection_type='polar'):
        super(ProjResNet, self).__init__()

        self.data_object = data_object
        self.transform = data_object.transform
        self.k = data_object.k
        self.d_hat = d_hat
        self.L = L
        self.type = projection_type
        assert self.type == 'polar' or self.type == 'qr'

        h = torch.Tensor(1)
        h[0] = 1 / L
        self.h = nn.Parameter(h, requires_grad=True)  # random stepsize
        K = len(data_object.labels_map)
        batch_size, n_matrices, self.d = data_object.all_data[0].shape  # [1500, 3, 28*k] if  svd
        if self.d_hat == 'none':
            self.d_hat = self.d

        assert self.d_hat >= self.d

        self.dim = int(self.d_hat / self.k)
        # L layers defined by affine operation z = Ky + b
        layers = [nn.Linear(self.d, self.d_hat, bias=False)]  # input layer with no bias
        self.d = int(np.sqrt(self.d))  # 28*28, 32*32
        for layer in range(L):
            layers.append(nn.Linear(self.d_hat, self.d_hat))
        self.layers = nn.ModuleList(layers)

        if self.transform == 'svd':
            # affine classifier
            self.classifier = nn.Linear(self.dim * self.dim, K)
        else:
            # affine classifier
            self.classifier = nn.Linear(self.d * self.d, K)
        # activation
        self.act = nn.ReLU()
        # hypothesis function to normalize prediction
        self.hyp = nn.Softmax(dim=1)

        self.layers[0].weight.requires_grad_(False)  # input layer not trained, i.e. weight is fixed

    def forward(self, X):
        """Propagate data through network.

        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).

        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output
            after applying classifier and hypothesis function). Shape
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape
            (num_samples, d_hat, L+1).
        """

        def projection(A, projection_type='polar'):
            """
            :param A: Matrix to be projected onto the Stiefel
            :param projection_type: Type of projection; polar or QR
            :return: A in Stiefel
            """

            if projection_type == 'polar':
                """
                D. Higham (1997)
                """
                U, S, Vh = svd(A)
                k = self.k
                n = A.shape[1]
                m = A.shape[2]
                # (U V*) (V S V*)
                return U[:, 0:n, 0:k] @ Vh[:, 0:k, 0:m]

            elif type == 'qr':
                """
                Dieci, Russell & van Vleck (1994).
                """
                Q, R = qr(A)
                return Q

        # track transformation of features
        X_transformed = torch.empty(X.shape[0], X.shape[1], self.d_hat, self.L + 1)  # [1500, 3, 28*k, 101]
        self.X_transformed = X_transformed
        k = self.k
        if self.transform == 'svd':
            self.integration_error = torch.empty(X.shape[0], 2, self.L + 1)
        else:
            self.integration_error = torch.empty(X.shape[0], self.L + 1)
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            # input layer with no activation
            if i == 0:
                if self.transform == 'polar':
                    X = layer(X)  # UP = polar
                    # Projection on to Stiefel
                    U_tilde = X[:, 0].unflatten(1, (self.d, self.d))
                    P = X[:, 1]
                    U = projection(U_tilde, self.transform)
                    X_new = torch.empty((X.shape[0], 2, self.d * self.d))
                    X_new[:, 0] = torch.flatten(U, 1, 2)
                    X_new[:, 1] = P
                    self.integration_error[:, i] = norm(U - U_tilde)
                    X = X_new

                elif self.transform == 'svd':
                    # time integration
                    X = layer(X)
                    # Projection on to Stiefel
                    U_tilde = X[:, 0].unflatten(1, (self.dim, k))
                    Vh_tilde = X[:, 2].unflatten(1, (k, self.dim))
                    V_tilde = transpose(Vh_tilde, 1, 2)  # comes out of propagated SVD, so this is (still) the transpose
                    S = X[:, 1]
                    U = projection(U_tilde, self.type)
                    V = projection(V_tilde, self.type)
                    self.integration_error[:, 0, i] = norm(U - U_tilde)
                    self.integration_error[:, 1, i] = norm(V - V_tilde)
                    X_new = torch.empty((X.shape[0], 3, self.dim * k))
                    X_new[:, 0] = torch.flatten(U, 1, 2)
                    X_new[:, 1] = S
                    X_new[:, 2] = torch.flatten(V, 1, 2)
                    X = X_new

                elif self.transform == 'qr':

                    # time integration
                    X = layer(X)
                    Q_tilde = X[:, 0].unflatten(1, (self.d, self.d))
                    # Projection on Stiefel
                    Q = projection(Q_tilde, self.transform)
                    self.integration_error[:, i] = norm(Q - Q_tilde)
                    X_new = torch.empty((X.shape[0], 2, self.d_hat))
                    X_new[:, 0] = torch.flatten(Q, 1, 2)
                    X_new[:, 1] = X[:, 1]
                    X = X_new

            # residual layers
            else:
                if self.transform == 'polar':

                    # time integration
                    X = X + self.h * self.act(layer(X))
                    # Projection on to Stiefel
                    U_tilde = X[:, 0].unflatten(1, (self.d, self.d))
                    P = X[:, 1]
                    U = projection(U_tilde, self.transform)
                    self.integration_error[:, i] = norm(U - U_tilde)
                    X_new = torch.empty((X.shape[0], 2, self.d * self.d))
                    X_new[:, 0] = torch.flatten(U, 1, 2)
                    X_new[:, 1] = P
                    X = X_new
                elif self.transform == 'svd':

                    # time integration
                    X = X + self.h * self.act(layer(X))
                    # Projection on to Stiefel
                    U_tilde = X[:, 0].unflatten(1, (self.dim, k))
                    V_tilde = X[:, 2].unflatten(1, (self.dim, k))
                    S = X[:, 1]
                    U = projection(U_tilde, self.type)
                    V = projection(V_tilde, self.type)
                    self.integration_error[:, 0, i] = norm(U - U_tilde)
                    self.integration_error[:, 1, i] = norm(V - V_tilde)
                    X_new = torch.empty((X.shape[0], 3, self.dim * k))
                    X_new[:, 0] = torch.flatten(U, 1, 2)
                    X_new[:, 1] = S
                    X_new[:, 2] = torch.flatten(V, 1, 2)
                    X = X_new
                elif self.transform == 'qr':

                    # time integration
                    X = X + self.h * self.act(layer(X))
                    Q_tilde = X[:, 0].unflatten(1, (self.d, self.d))
                    # Projection on Stiefel
                    Q = projection(Q_tilde, self.transform)
                    self.integration_error[:, i] = norm(Q - Q_tilde)
                    X_new = torch.empty((X.shape[0], 2, self.d_hat))
                    X_new[:, 0] = torch.flatten(Q, 1, 2)
                    X_new[:, 1] = X[:, 1]
                    X = X_new

            X_transformed[:, :, :, i] = X

        # maybe this does not make sense.. ? Multiplying these back together

        if self.transform == 'svd':
            U = X[:, 0].unflatten(1, (self.dim, k))
            S = X[:, 1].unflatten(1, (self.dim, k))
            S = S[:, 0:k, 0:k]
            V = X[:, 2].unflatten(1, (self.dim, k))
            X = U @ S @ torch.transpose(V, 1, 2)
            X = torch.flatten(X, 1, 2)

        elif self.transform == 'polar':
            U = X[:, 0].unflatten(1, (self.d, self.d))
            P = X[:, 1].unflatten(1, (self.d, self.d))
            X = U @ P
            X = torch.flatten(X, 1, 2)

        elif self.transform == 'qr':
            Q = X[:, 0].unflatten(1, (self.d, self.d))
            R = X[:, 1].unflatten(1, (self.d, self.d))
            X = Q @ R
            X = torch.flatten(X, 1, 2)

        # apply classifier
        X_classified = self.classifier(X)  # Projects dimension of network down to output space. Linear classifier

        # apply hypothesis function
        X_predicted = self.hyp(X_classified)  # Maps a batch into a probability distribution

        return [X_predicted, X_classified, X_transformed]

    @property
    def orthogonality(self):
        # Investigating orthogonality of matrices through the network
        if self.transform == 'svd':  # [1500, 3, 28*k, 101]
            UsL = self.X_transformed[:, 0, :, :]
            VsL = self.X_transformed[:, 2, :, :]
            # print("UsL", UsL.shape)
            L = self.L
            k = self.k
            err_U = np.zeros((UsL.shape[0], L + 1))
            err_V = np.zeros((VsL.shape[0], L + 1))
            for i, u_evol in enumerate(UsL):
                v_evol = VsL[i]  # pick a matrix
                for layer in range(L + 1):
                    u = u_evol[:, layer]
                    v = v_evol[:, layer]
                    err_U[i, layer] = check_orthogonality(u.unflatten(0, (self.dim, k)))
                    err_V[i, layer] = check_orthogonality(v.unflatten(0, (self.dim, k)))
            # print(err_U.shape)
            err_U = np.average(err_U, axis=0)
            err_V = np.average(err_V, axis=0)
            # print(err.shape)

            fig, ax = plt.subplots(2)
            plt.tight_layout()

            ax[0].set_title(r' $|| I - U^T U ||_F$')
            ax[0].plot(err_U)
            ax[0].set_xlabel("layer of network")
            ax[0].set_ylabel("error")

            ax[1].set_title(r' $|| I - V^T V ||_F$')
            ax[1].plot(err_V)
            ax[1].set_xlabel("layer of network")
            ax[1].set_ylabel("error")
            plt.show()

            s = ' highest average orthogonality error in U: ' + str(max(err_U)) + \
                '\n highest average orthogonality error in V: ' + str(max(err_V))

            return s, err_U, err_V

        else:  # [1500, 3, 28*k, 101]
            UsL = self.X_transformed[:, 0, :, :]
            # print("UsL", UsL.shape)
            L = self.L
            err_U = np.zeros((UsL.shape[0], L + 1))
            for i, u_evol in enumerate(UsL):
                for layer in range(L + 1):
                    u = u_evol[:, layer]
                    err_U[i, layer] = check_orthogonality(u.unflatten(0, (self.d, self.d)))

            # print(err_U.shape)
            err_U = np.average(err_U, axis=0)
            # print(err.shape)
            plt.tight_layout()

            plt.title(r' $|| I - U^T U ||_F$')
            plt.plot(err_U)
            plt.xlabel("layer of network")
            plt.ylabel("average error")
            plt.show()
            s = ' highest average orthogonality error in U: ' + str(max(err_U))
            return s, err_U

    @property
    def net_structure(self):
        print('network architecture:')
        print(self, '\n')

        # total number of neurons
        num_neurons = self.d + self.d_hat * self.L
        print('total number of neurons in network:              ', num_neurons)

        # total number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total number of parameters in network:           ', num_params)
        print('total number of trainable parameters in network: ', num_trainable_params)

        s = 'This is a Projection network: \n' \
            ' depth: L = ' + str(self.L) + '\n' \
            ' activation function: ' + str(self.act) + '\n' \
            ' number of trainable parameters: ' + str(num_trainable_params)
        return s

    @property
    def plot_integration_error(self):
        # want to track how bad the numerical integration projects away from the Stiefel manifold
        # Could maybe be used to reject timesteps, and try with a smaller step h, but still we
        # have very limited control of the error in each step
        if self.transform == 'svd':
            err_U = np.average(self.integration_error[:, 0, :], axis=0)
            err_V = np.average(self.integration_error[:, 1, :], axis=0)
            plt.tight_layout()
            plt.title(r' Integration error')
            plt.plot(err_U, label=r'$|| U - \tilde U ||_F$')
            plt.plot(err_V, label=r'$|| V - \tilde V ||_F$')
            plt.xlabel("layer of network")
            plt.ylabel(" average error")
            plt.show()

            return err_U, err_V

        else:
            err = np.average(self.integration_error[:, :], axis=0)
            plt.tight_layout()
            plt.title(r' Integration error')
            plt.plot(err, label=r'$|| U - \tilde U ||_F$')
            plt.xlabel("layer of network")
            plt.ylabel(" average error")
            plt.show()

            return err

    @property
    def rank_evolution(self):
        if self.transform == 'svd':
            UsL = self.X_transformed[:, 0, :, :]
            SsL = self.X_transformed[:, 1, :, :]
            VsL = self.X_transformed[:, 2, :, :]

            L = self.L
            k = self.k
            ranks = np.zeros((UsL.shape[0], L + 1))
            for layer in range(L + 1):
                u = UsL[:, :, layer].unflatten(1, (self.dim, k))
                s = SsL[:, :, layer].unflatten(1, (self.dim, k))
                s = s[:, 0:k, 0:k]
                v = VsL[:, :, layer].unflatten(1, (self.dim, k))
                x = u @ s @ torch.transpose(v, 1, 2)
                ranks[:, layer] = matrix_rank(x)

        elif self.transform == 'polar':
            WsL = self.X_transformed[:, 0, :, :]
            PsL = self.X_transformed[:, 1, :, :]

            L = self.L
            ranks = np.zeros((WsL.shape[0], L + 1))
            for layer in range(L + 1):
                w = WsL[:, :, layer].unflatten(1, (self.d, self.d))
                p = PsL[:, :, layer].unflatten(1, (self.d, self.d))
                x = w @ p
                ranks[:, layer] = matrix_rank(x)

        else:
            QsL = self.X_transformed[:, 0, :, :]
            RsL = self.X_transformed[:, 1, :, :]
            L = self.L
            ranks = np.zeros((QsL.shape[0], L + 1))
            for layer in range(L + 1):
                q = QsL[:, :, layer].unflatten(1, (self.d, self.d))
                r = RsL[:, :, layer].unflatten(1, (self.d, self.d))
                x = q @ r
                ranks[:, layer] = matrix_rank(x)

        # print(err_U.shape)
        ranks = np.average(ranks, axis=0)
        plt.tight_layout()
        plt.title(r' $rank (X)$')
        plt.plot(ranks)
        plt.xlabel("layer of network")
        plt.ylabel("average rank")
        plt.show()
        return ranks
