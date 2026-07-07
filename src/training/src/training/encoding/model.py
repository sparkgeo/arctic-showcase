from pathlib import Path

import torch
from claymodel.module import ClayMAEModule


def select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_clay_module(
    checkpoint_path: Path, metadata_path: Path, device: torch.device | None = None
) -> ClayMAEModule:
    # model_size="large" is mandatory: the default "base" is 768-dim and silently
    # breaks the feature contract's 1024-dim assumption.
    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_size="large",
        mask_ratio=0.0,
        shuffle=False,
        metadata_path=str(metadata_path),
    )
    module.eval()
    return module.to(device or select_device())
