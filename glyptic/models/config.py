from typing import Any


def get_config() -> dict[str, Any]:
    config = dict(
        # TODO: Change params to more appropriate values
        rng_seed=42,
        learning_rate=0.02,
        batch_size=128,
        num_steps=1000,
        num_epochs=5,
        image_height=32,
        image_width=48,
        initial_channels=64,
        # Currently insufficient memory for (1, 2, 2, 4)
        channel_multipliers=(1, 2, 2, 2),
        blocks_with_attention=(False, False, True, True),
        num_blocks=2,
    )
    return config
