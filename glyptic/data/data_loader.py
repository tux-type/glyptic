from pathlib import Path
import random

from PIL import Image
import numpy as np


class FrameKeyDataLoader:
    def __init__(self, path: str, batch_size: int = 1, shuffle: bool = False):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: Improve efficiency by reusing loaded images (only load once)
        self.input_image_paths, self.label_image_paths = self._get_image_paths()
        assert len(self.input_image_paths) == len(self.label_image_paths), (
            f"number of input samples ({len(self.input_image_paths)})"
            " does not match number of label samples ({len(self.label_image_paths)})"
        )
        self.num_samples = len(self.input_image_paths)

    def _get_image_paths(self):
        all_image_paths = list(Path(self.path).rglob("*.png"))

        # Pairing up consecutive imgaes into (input, label) results in (n - 1) samples.
        input_image_paths = all_image_paths[:-1]
        label_image_paths = all_image_paths[1:]
        return input_image_paths, label_image_paths

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img)

    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[start_idx : (start_idx + self.batch_size)]
            batch_input_images = [
                self._load_image(self.input_image_paths[i]) for i in batch_indices
            ]
            batch_label_images = [
                self._load_image(self.label_image_paths[i]) for i in batch_indices
            ]
            yield np.array(batch_input_images), np.array(batch_label_images)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
