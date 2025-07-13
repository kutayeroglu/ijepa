import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses,
    val_losses,
    num_epochs,
    save_path="training_loss_plot.png",
):
    """
    Plots and saves the training and validation loss curves.
    """
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label="Training Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()
