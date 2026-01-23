import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


# ================================================================
# Saving variables
# ================================================================
def linear_schedule(initial_value, final_value, final_step):
    """Linear schedule for exploration rate.

    Parameters
    ----------
    initial_value: float
        Initial value of the schedule
    final_value: float
        Final value of the schedule
    final_step: int
        Step at which the schedule reaches its final value

    Returns
    -------
    schedule: function
        A function that takes a step and returns the current value
    """

    def schedule(step):
        """Returns the current value of the schedule."""
        if step >= final_step:
            return final_value
        return initial_value + (final_value - initial_value) * step / final_step

    return schedule


def load_state(path):
    """Load model state from file.
    
    Parameters
    ----------
    path: str
        Path to the saved model state
    """
    return torch.load(path)


def save_state(path):
    """Save model state to file.
    
    Parameters
    ----------
    path: str
        Path where to save the model state
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(path)


# ================================================================
# Input processing
# ================================================================

class Processor:
    """Base class for input processors."""

    def __init__(self, name="(unnamed)"):
        """Initialize a processor with a name."""
        self.name = name

    def process(self, data):
        """Process input data.
        
        Parameters
        ----------
        data: Any
            Input data to process
            
        Returns
        -------
        processed_data: Any
            Processed data
        """
        raise NotImplementedError()


class BatchProcessor(Processor):
    def __init__(self, shape, device='cpu', dtype=torch.float32, name=None):
        """Creates a processor for a batch of tensors of a given shape and dtype
        
        Parameters
        ----------
        shape: tuple of int
            Shape of a single element of the batch
        device: str or torch.device
            Device to place tensor on
        dtype: torch.dtype
            Number representation used for tensor contents
        name: str
            Name of the processor
        """
        super().__init__(name=name if name is not None else "BatchProcessor")
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def process(self, data):
        """Convert data to tensor with appropriate shape and type.
        
        Parameters
        ----------
        data: numpy.ndarray
            Input data
            
        Returns
        -------
        processed_data: torch.Tensor
            Processed data as tensor
        """
        return torch.tensor(data, dtype=self.dtype, device=self.device)


class Uint8Processor(BatchProcessor):
    def __init__(self, shape, device='cpu', name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.
        
        Parameters
        ----------
        shape: tuple of int
            Shape of a single element
        device: str or torch.device
            Device to place tensor on
        name: str
            Name of the processor
        """
        super().__init__(shape, device, torch.float32, name=name if name is not None else "Uint8Processor")

    def process(self, data):
        """Convert uint8 data to normalized float32 tensor.
        
        Parameters
        ----------
        data: numpy.ndarray
            Input data in uint8 format
            
        Returns
        -------
        processed_data: torch.Tensor
            Processed data as normalized float32 tensor
        """
        return torch.tensor(data, device=self.device, dtype=torch.uint8).float() / 255.0


def plot_gaussian_contours(gaussians, grid_size=200, contour_levels=5, colors=None, filled=False, saved_path=None):
    """
    Plot contours for multiple 2D Gaussian distributions.

    Parameters
    ----------
    gaussians : list of (mean, cov)
        mean : np.ndarray of shape (2,)
        cov : np.ndarray of shape (2, 2)
    grid_size : int
        Number of grid points per axis for PDF evaluation.
    contour_levels : int
        Number of contour levels to plot.
    colors : list of str or None
        Colors for each Gaussian. If None, defaults to matplotlib cycle.
    filled : bool
        If True, use filled contours with transparency; else line contours.
    """
    # Compute global bounds from means
    gaussians_2d = [(mean[:2], cov[:2, :2]) for mean, cov in gaussians]

    # Determine bounds
    all_means = np.array([m for m, _ in gaussians_2d])
    # x_min, x_max = all_means[:, 0].min() - 3, all_means[:, 0].max() + 3
    # y_min, y_max = all_means[:, 1].min() - 3, all_means[:, 1].max() + 3
    x_min, x_max = 0, 50
    y_min, y_max = -10, 10

    # Create grid
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Colors
    if colors is None:
        colors = plt.cm.tab10.colors

    plt.figure(figsize=(8, 6))
    for i, (mean, cov) in enumerate(gaussians_2d):
        rv = multivariate_normal(mean, cov)
        Z = rv.pdf(pos)
        color = colors[i % len(colors)]
        if filled:
            plt.contourf(X, Y, Z, levels=contour_levels, alpha=0.5, colors=[color])
        else:
            plt.contour(X, Y, Z, levels=contour_levels, colors=[color])
        plt.scatter(*mean, color=color, marker='x', s=100, label=f"Mean {mean}")

    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.legend()
    plt.title('First Two Dimensions of Gaussians')
    plt.grid(True)
    plt.axis('equal')
    if saved_path is not None:
        plt.savefig(saved_path)
        plt.close()
    else:
        plt.show()


def plot_q_values_heatmap(q_values, saved_path, timesteps=None, action_labels=None, fmt=".2f"):
    """
    Plot a heatmap of Q-values over time with text annotations.

    Parameters:
        q_values (ndarray): shape (num_steps, num_actions), Q-values.
        timesteps (list or ndarray): optional, labels for x-axis.
        action_labels (list): optional, labels for y-axis (must match num_actions).
        fmt (str): format for text values inside cells.
    """
    q_values = np.array(q_values)  # ensure ndarray
    num_steps, num_actions = q_values.shape

    if timesteps is None:
        timesteps = np.arange(num_steps)
    if action_labels is None:
        action_labels = [f"Action {i}" for i in range(num_actions)]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(q_values.T, aspect='auto', origin='lower',
                   extent=[timesteps[0], timesteps[-1], 0, num_actions],
                   cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Q-value")

    # Axis labels
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Actions")
    ax.set_yticks(np.arange(num_actions) + 0.5)
    ax.set_yticklabels(action_labels)
    ax.set_title("Q-values Heatmap")

    # Add text annotations
    if len(timesteps)<=5:
        for i, t in enumerate(timesteps):
            max_action = np.argmax(q_values[i])
            for j in range(num_actions):
                if t == 0:
                    ax.text(t + 0.25, j + 0.5, format(q_values[i, j], fmt),
                            ha="center", va="center", color="white")
                elif t == len(timesteps) - 1:
                    ax.text(t - 0.25, j + 0.5, format(q_values[i, j], fmt),
                            ha="center", va="center", color="white")
                else:
                    ax.text(t, j + 0.5, format(q_values[i, j], fmt),
                            ha="center", va="center", color="white")
            if t == 0:
                ax.text(t + 0.45, max_action + 0.5, "★", ha="center", va="center",
                        color="red", fontsize=14, fontweight="bold")
            elif t == len(timesteps) - 1:
                ax.text(t - 0.1, max_action + 0.5, "★", ha="center", va="center",
                        color="red", fontsize=14, fontweight="bold")
            else:
                ax.text(t+0.15, max_action + 0.5, "★", ha="center", va="center",
                        color="red", fontsize=14, fontweight="bold")
    plt.savefig(saved_path)
    plt.close()
