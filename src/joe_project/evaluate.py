import matplotlib.pyplot as plt
import torch
import typer
from model import S1model

from data import corrupt_mnist

# The model and data are moved to GPU or Apple MPS accelerator if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""

    # Printing the latest checkpoint from the training of the model
    print(model_checkpoint)

    # Loading the model
    model = S1model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))

    # Splitting the test data into batches
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Turning dropout off
    model.eval()

    # initializing counts
    correct, total = 0, 0

    # Predicting and saving the number of correct predictions
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
