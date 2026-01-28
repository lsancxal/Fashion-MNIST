"""Training visualization for Fashion-MNIST."""

import matplotlib.pyplot as plt


def plot_training_results(cost_list, accuracy_list, show=True):
    """
    Plot training cost and accuracy over epochs.
    
    Args:
        cost_list: List of cost values per epoch.
        accuracy_list: List of accuracy values per epoch.
        show: Whether to call plt.show().
        
    Returns:
        fig: The matplotlib figure.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot cost on primary axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost', color='tab:red')
    ax1.plot(cost_list, color='tab:red', label='Cost')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot accuracy on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(accuracy_list, color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title('Training Progress')
    
    if show:
        plt.show()
    
    return fig
