import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform, color
from scipy import ndimage as ndi

def plot_heatmap(heatmap, original, ax=None, cmap='RdBu_r', percentile=99, dilation=0.5, alpha=0.25):
    if ax is None:
        _, ax = plt.subplots()

    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, axis=0)

    original_height, original_width = original.shape[:2]

    extent = (0, original_width, 0, original_height)

    cmap_original = plt.get_cmap('Greys_r')
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original_height is not None and original_width is not None:
        original_greyscale = original.mean(-1) if len(original.shape) > 2 else original
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max

    ax.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=alpha)

import torch


def generate_heatmap_pytorch(model, input_tensor, target_label, num_steps, epsilon=1e-2):
    """
    Generate heatmap using guided backpropagation for a PyTorch model.

    Parameters:
    - model: PyTorch model
    - input_tensor: Input tensor (image) for which to generate the heatmap
    - target_label: Target class label for the heatmap
    - num_steps: Number of steps for the approximation
    - epsilon: Perturbation value for approximation

    Returns:
    - heatmap: Generated heatmap
    """
    model.eval()

    # Initialize the gradients
    input_tensor.requires_grad = True
    model.zero_grad()

    # Forward pass
    logits = model(input_tensor)
    loss = logits[0, target_label]  # Assuming single-label classification
    loss.backward()

    # Get gradients and convert to numpy
    gradients = input_tensor.grad.data.numpy()[0]
    print(gradients.shape)

    # Approximate integrated gradients
    integrated_gradients = (input_tensor.data.numpy()[0] - input_tensor.data.numpy()[0].mean(axis=1, keepdims=True)) * gradients

    # Calculate the average over num_steps
    for step in range(1, num_steps + 1):
        step_ratio = float(step) / num_steps
        step_image = input_tensor.data.numpy()[0] + step_ratio * epsilon * np.sign(gradients)
        step_image = torch.from_numpy(step_image).float().unsqueeze(0)
        step_image.requires_grad = True

        step_logits = model(step_image)
        step_loss = step_logits[0, target_label]  # Assuming single-label classification
        step_loss.backward()

        step_gradients = step_image.grad.data.numpy()[0]
        integrated_gradients += (step_image.data.numpy()[0] - input_tensor.data.numpy()[0].mean(axis=1, keepdims=True)) * step_gradients

    # Average the gradients
    avg_gradients = integrated_gradients / num_steps


    # Generate the heatmap by taking the absolute value of the average gradients
    heatmap = np.abs(avg_gradients).max(axis=0)

    return heatmap

