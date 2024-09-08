from pathlib import Path

import pytest

from glyptic.models import get_config, train_and_evaluate


@pytest.fixture(scope="module")
def data_dir():
    return str(Path(__file__).parent.parent) + "/dataset/collection_20240908-153517"


def test_train_and_evaluate(data_dir):
    config = get_config()
    config["batch_size"] = 32
    config["num_epochs"] = 1
    train_and_evaluate(config=config, data_dir=data_dir)
