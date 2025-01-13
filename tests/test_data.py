from src.joe_project.data import corrupt_mnist
import pytest
import os.path
from tests.__init__ import _PATH_DATA
import torch

# if data is not there
@pytest.mark.skipif(not os.path.exists("joe_project/data"), reason="Data files not found")


def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "train data not of length 30000"                             # train set should contain 30000 points
    assert len(test) == 5000, "test data not of length 5000"                                 # test set should contain 5000 points
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "x in data point not a (1,28,28) tensor"                  # x should be a tensor of (1, 28, 28)
            assert y in range(10), "y in data point not an int range 0 to 9"                         # y should be int in range 0 to 9
    train_targets = torch.unique(train.tensors[1])                                                   # extract labels in train set
    assert (train_targets == torch.arange(0,10)).all(), "train labels do not cover range 0 to 9"     # check they cover all the range of 0 to 9
    test_targets = torch.unique(test.tensors[1])                                                     # extract labels in test set
    assert (test_targets == torch.arange(0,10)).all(), "test labels do not cover range 0 to 9"       # check they cover all the range of 0 to 9
