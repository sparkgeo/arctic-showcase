import einops
import numpy as np
import torch
from claymodel.module import ClayMAEModule
from numpy.typing import NDArray

from training.data_loader.chip import Chip
from training.encoding.metadata import SarMetadata


def encode_chip(
    chip: Chip, module: ClayMAEModule, sar_meta: SarMetadata, device: torch.device
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Encode one chip with Clay's frozen encoder (mask_ratio=0.0, shuffle=False).

    Returns (patch_tokens, cls_token) as numpy arrays:
    patch_tokens (1, 1024, 32, 32), cls_token (1, 1024).
    """
    sar_mean = torch.tensor(sar_meta.mean, dtype=torch.float32)
    sar_std = torch.tensor(sar_meta.std, dtype=torch.float32)
    sar_norm = (torch.tensor(chip.sar, dtype=torch.float32) - sar_mean[:, None, None]) / sar_std[
        :, None, None
    ]

    datacube = {
        "pixels": sar_norm.unsqueeze(0).to(device),
        "waves": torch.tensor(sar_meta.wavelengths, dtype=torch.float32).to(device),
        "gsd": torch.tensor(sar_meta.gsd, dtype=torch.float32).to(device),
        "time": torch.tensor(chip.time_encoding, dtype=torch.float32).unsqueeze(0).to(device),
        "latlon": torch.tensor(chip.latlon_encoding, dtype=torch.float32).unsqueeze(0).to(device),
    }

    with torch.no_grad():
        encoded, *_ = module.model.encoder(datacube)

    cls_token: NDArray[np.float32] = encoded[:, 0, :].cpu().numpy()
    patch_tokens: NDArray[np.float32] = einops.rearrange(
        encoded[:, 1:, :].cpu().numpy(), "b (h w) d -> b d h w", h=32, w=32
    )

    return patch_tokens, cls_token
