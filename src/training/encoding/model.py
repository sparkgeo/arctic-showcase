from pathlib import Path
from typing import Any

import claymodel.model as clay_model
import torch
from claymodel.module import ClayMAEModule


def select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


_original_create_model = clay_model.timm.create_model


def _create_teacher_without_download(*args: Any, **kwargs: Any) -> Any:
    # ClayMAE.__init__ unconditionally downloads pretrained teacher weights from
    # HF Hub for a distillation loss we never use (we only call model.encoder()).
    # The subsequent load_from_checkpoint() overwrites these params anyway, so the
    # download is pure wasted network dependency at feature-encoding time.
    kwargs["pretrained"] = False
    return _original_create_model(*args, **kwargs)


clay_model.timm.create_model = _create_teacher_without_download


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
