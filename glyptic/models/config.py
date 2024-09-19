from typing import Any


def get_config() -> dict[str, Any]:
    config = dict(
        rng_seed=42,
        learning_rate=0.00001,
        batch_size=32,
        num_steps=1000,
        num_epochs=3,
        image_height=32,
        image_width=48,
        initial_channels=64,
        # Currently insufficient memory for (1, 2, 2, 4)
        channel_multipliers=(1, 2, 2, 2),
        blocks_with_attention=(False, False, True, True),
        num_blocks=2,
        dropout_rate=0.1,
        clip_max_norm=1.0,
        train_data_dir="data/training/collection_20240908-153517",
        val_data_dir="data/validation/collection_20240914-154513",
    )
    return config
