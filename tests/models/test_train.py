from pathlib import Path

import pytest

from glyptic.models import get_config, train_and_evaluate


@pytest.fixture(scope="module")
def train_data_dir():
    return str(Path(__file__).parent.parent) + "/dataset/training/collection_20240908-153517"


@pytest.fixture(scope="module")
def val_data_dir():
    return str(Path(__file__).parent.parent) + "/dataset/validation/collection_20240908-153517"


def test_train_and_evaluate(train_data_dir: str, val_data_dir: str):
    config = get_config()
    config["batch_size"] = 32
    config["num_epochs"] = 1
    config["train_data_dir"] = train_data_dir
    config["val_data_dir"] = val_data_dir
    train_and_evaluate(config=config)
