import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

import prediction
from data import restore_svd, from_tucker_decomposition, svd_transform, tucker_decomposition
from tensorly import tucker_to_tensor

def fgsm_attack(image, epsilon, data_grad, k = 28, transform = "none"):
    if transform != "none": assert k != 28
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    if transform == "svd":
        image = restore_svd(image[0], k) #unwrap
        sign_data_grad = restore_svd(sign_data_grad[0], k)
    elif transform == "tucker":
        image = tucker_to_tensor(from_tucker_decomposition(image[0], k)) #unwrap
        sign_data_grad = tucker_to_tensor(from_tucker_decomposition(sign_data_grad[0], k))

    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    if transform == "svd":
        perturbed_image = svd_transform(perturbed_image, k)
        perturbed_image = torch.unsqueeze(perturbed_image, 0)
    elif transform == "tucker":
        perturbed_image = tucker_decomposition(perturbed_image, k)
        perturbed_image = torch.unsqueeze(perturbed_image, 0)
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    transform = model.data_object.transform
    k = model.k

    # Loop over all examples in test set
    data_batch, target_batch = next(iter(test_loader))

    for data, target in zip(data_batch, target_batch):
        batch_size = 1
        data = torch.unsqueeze(data, 0)
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        target = ((target == 1).nonzero(as_tuple=True)[0])

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        X_predicted, _, _ = model(data)

        init_pred = prediction.pred(X_predicted) # from probabilities to classified
        init_pred = init_pred[0] #unwrap
        init_pred = ((init_pred == 1).nonzero(as_tuple=True)[0])

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(X_predicted, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        if epsilon != 0:
            perturbed_data = fgsm_attack(data, epsilon, data_grad, k=k, transform=transform)
        else: perturbed_data = data
        # Re-classify the perturbed image
        X_predicted, _, _ = model(perturbed_data)
        final_pred = prediction.pred(X_predicted)
        final_pred = final_pred[0] #unwrap
        final_pred = ((final_pred == 1).nonzero(as_tuple=True)[0])
        # Check for success
        #print("init", init_pred.item())
        #print("final", final_pred.item())
        #print("truth", target.item())
        #print("\n")
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(data_batch))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(data_batch), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# Plot several examples of adversarial samples at each epsilon
def plot_adversarial_examples(epsilons, examples, k=28, transform=None ):
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(r'$\epsilon = ${}'.format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            if transform == 'svd':
                ex = restore_svd(ex, k)
            elif transform == 'tucker':
                ex = tucker_to_tensor( from_tucker_decomposition(ex, k))
            else:
                flattened_size = ex.shape[0]
                n = int(np.sqrt(flattened_size))
                ex = ex.reshape((n,n))
            plt.imshow(ex)
    plt.tight_layout()

def load_model(net, data_object, L, path, use_cuda=True):
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # Initialize the network
    model = net(data_object, L).to(device)
    # Load the pretrained model
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()
    return model

def run_attack(model, epsilons, use_cuda=True):
    accuracies = []
    examples = []
    test_loader = model.data_object.test_dataloader
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    return accuracies, examples