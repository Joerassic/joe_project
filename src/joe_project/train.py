import matplotlib.pyplot as plt
import torch
import typer
import wandb
from model import S1model
from data import corrupt_mnist

# The model and data are moved to GPU or Apple MPS accelerator if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-4, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""

    # printing hyperparameters
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # wandb logging
    wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    # Loading the model
    model = S1model().to(DEVICE)

    # Splitting the training data into batches
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Using cross entropy as loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Using SGD as optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Calculating loss and accuracy
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()


            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            # printing the loss every 100th iteration
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    # Saving the model!
    torch.save(model.state_dict(), "models/model.pth")

    # Plotting the loss and accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

if __name__ == "__main__":
    typer.run(train)
