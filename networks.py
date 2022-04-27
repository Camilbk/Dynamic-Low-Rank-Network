import torch
import torch.nn as nn
from torch import eye, transpose, inverse
from torch.linalg import norm, svd, qr
import numpy as np
from matplotlib import pyplot as plt
from torch.linalg import matrix_rank
import prediction


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
    inv = torch.inverse(Id - 0.5 * DT @ C)
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
            s_inv = inverse(s)
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
