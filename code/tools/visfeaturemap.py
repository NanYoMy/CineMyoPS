
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.detach().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def visualize_feat(input_tensor,transformed_input_tensor):

    input_tensor=input_tensor.cpu()[0].unsqueeze(1)
    transformed_input_tensor=transformed_input_tensor.cpu()[0].unsqueeze(1)

    # ori=torchvision.utils.make_grid(input_tensor)
    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))

    # affine=torchvision.utils.make_grid(transformed_input_tensor)
    out_grid = convert_image_np(
        torchvision.utils.make_grid(transformed_input_tensor))

    # Plot the results side-by-side
    # fig = plt.figure(1)
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(in_grid)
    # axarr[0].set_title('Dataset Images')

    # axarr[1].imshow(out_grid)
    # axarr[1].set_title('Transformed Images')

    return in_grid,out_grid