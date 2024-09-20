from pathlib import Path
import random

from PIL import Image
import numpy as np


class FrameKeyDataLoader:
    def __init__(
        self,
        path: str,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last=False,
        seed: int | None = None,
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle and seed is not None:
            random.seed(seed)

        # TODO: Improve efficiency by reusing loaded images (only load once)
        self.input_image_paths, self.label_image_paths = self._get_image_paths()
        assert len(self.input_image_paths) == len(self.label_image_paths), (
            f"number of input samples ({len(self.input_image_paths)})"
            " does not match number of label samples ({len(self.label_image_paths)})"
        )
        if drop_last:
            leftover = len(self.input_image_paths) % self.batch_size
            self.input_image_paths = self.input_image_paths[:-leftover]
            self.label_image_paths = self.label_image_paths[:-leftover]
        self.num_samples = len(self.input_image_paths)
        # Lazy init
        self._all_images: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def all_images(self):
        if self._all_images is None:
            self._all_images = self._load_all()
        return self._all_images

    def _load_all(self) -> tuple[np.ndarray, np.ndarray]:
        input_images = np.array([self._load_image(img_path) for img_path in self.input_image_paths])
        label_images = np.array([self._load_image(img_path) for img_path in self.label_image_paths])
        return input_images, label_images

    def _get_image_paths(self):
        all_image_paths = sorted(list(Path(self.path).rglob("*.png")))

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
