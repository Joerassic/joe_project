from joe_project.model import S1model
import pytest
import torch


def test_model():
    model = S1model()
    x = torch.randn(1, 1, 28, 28)   # random input in the shape of the data
    y = model(x)                    # evaluating the model on the input
    assert y.shape == (1, 10), "output is not in shape (1,10)"


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_batch_size(batch_size: int) -> None:
    model = S1model()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10), "batch size not 32 or 64"
