from pathlib import Path

import pytest
import numpy as np

from glyptic.data import FrameKeyDataLoader


@pytest.fixture(scope="module")
def data_dir():
    return str(Path(__file__).parent.parent) + "/dataset/training/collection_20240908-153517"


def test_constructor(data_dir):
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=1, shuffle=False)
    assert data_loader.num_samples == (3000 - 1)
    assert len(data_loader.input_image_paths) == data_loader.num_samples
    assert len(data_loader.input_image_paths) == len(data_loader.label_image_paths)
    assert all([isinstance(image_path, Path) for image_path in data_loader.input_image_paths])
    assert all([isinstance(image_path, Path) for image_path in data_loader.label_image_paths])
    assert all([image_path.suffix == ".png" for image_path in data_loader.input_image_paths])
    assert all([image_path.suffix == ".png" for image_path in data_loader.label_image_paths])
    assert [
        input_path != label_path
        for input_path, label_path in zip(
            data_loader.input_image_paths, data_loader.label_image_paths
        )
    ]


def test_offset(data_dir):
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=1, shuffle=False)
    previous_batch_outputs = None
    for batch_inputs, batch_outputs in data_loader:
        if previous_batch_outputs is not None:
            assert np.array_equal(batch_inputs, previous_batch_outputs)
        previous_batch_outputs = batch_outputs


def test_num_batches(data_dir):
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=1, shuffle=False)
    assert len(data_loader) == data_loader.num_samples


def test_batch_single(data_dir):
    batch_size = 1
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=batch_size, shuffle=False)
    assert all(
        [
            isinstance(batch_inputs, np.ndarray) and isinstance(batch_labels, np.ndarray)
            for batch_inputs, batch_labels in data_loader
        ]
    )
    unique_batch_shapes = set(
        [(batch_inputs.shape, batch_labels.shape) for batch_inputs, batch_labels in data_loader]
    )
    assert len(unique_batch_shapes) == 1
    assert all(
        [batch_inputs.shape == batch_labels.shape for batch_inputs, batch_labels in data_loader]
    )
    assert all(
        [
            batch_inputs.shape[0] == batch_size and batch_labels.shape[0] == batch_size
            for batch_inputs, batch_labels in data_loader
        ]
    )


def test_batch_multiple(data_dir):
    # TODO: Consider adding a range/random/fuzzing for batch size
    batch_size = 20
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=batch_size, shuffle=False)
    assert all(
        [
            isinstance(batch_inputs, np.ndarray) and isinstance(batch_labels, np.ndarray)
            for batch_inputs, batch_labels in data_loader
        ]
    )
    batch_shapes = [
        (batch_inputs.shape, batch_labels.shape) for batch_inputs, batch_labels in data_loader
    ]
    assert [
        batch_input_shape == batch_label_shape
        for batch_input_shape, batch_label_shape in batch_shapes
    ]
    assert len(set(batch_shapes)) == (
        np.ceil(data_loader.num_samples % batch_size / batch_size) + 1
    ), "all shapes within each batch should be the same unless there is a trailing batch"
    assert [
        batch_input_shape[0] == batch_size and batch_label_shape[0] == batch_size
        for batch_input_shape, batch_label_shape in batch_shapes[:-1]
    ], "incorrect last batch size"
    assert batch_shapes[-1][0][0] == data_loader.num_samples - (
        (len(batch_shapes) - 1) * batch_size
    )
    assert (
        sum([batch_input_shape[0] for batch_input_shape, _ in batch_shapes])
        == data_loader.num_samples
    )


def test_shuffle(data_dir):
    batch_size = 20
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=batch_size, shuffle=True)

    epochs = 5
    all_epochs = []
    for _ in range(epochs):
        epoch_means = []
        for batch_inputs, _ in data_loader:
            epoch_means.extend([np.mean(batch_input) for batch_input in batch_inputs])
        all_epochs.append(epoch_means)

    assert any(all_epochs[i] != all_epochs[j] for i in range(epochs) for j in range(i + 1, epochs))


def test_no_shuffle(data_dir):
    batch_size = 20
    data_loader = FrameKeyDataLoader(path=data_dir, batch_size=batch_size, shuffle=False)

    epochs = 5
    all_epochs = []
    for _ in range(epochs):
        epoch_means = []
        for batch_inputs, _ in data_loader:
            epoch_means.extend([np.mean(batch_input) for batch_input in batch_inputs])
        all_epochs.append(epoch_means)

    assert all(all_epochs[i] == all_epochs[j] for i in range(epochs) for j in range(i + 1, epochs))


def test_drop_last(data_dir):
    expected_num_samples = 3000 - 1
    batch_size = 20
    data_loader = FrameKeyDataLoader(
        path=data_dir, batch_size=batch_size, shuffle=False, drop_last=True
    )
    assert expected_num_samples % batch_size != 0  # Check in case test parameters get changed
    assert data_loader.num_samples == (expected_num_samples - (expected_num_samples % batch_size))


def test_load_all(data_dir):
    batch_size = 20
    seed = 123
    data_loader_iterative_load = FrameKeyDataLoader(
        path=data_dir,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        load_all=False,
    )
    data_loader_load_all = FrameKeyDataLoader(
        path=data_dir,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        load_all=True,
    )

    for (x_it, y_it), (x_all, y_all) in zip(data_loader_iterative_load, data_loader_load_all):
        assert np.array_equal(x_it, x_all)
        assert np.array_equal(y_it, y_all)
