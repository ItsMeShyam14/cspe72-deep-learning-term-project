"""OSNet person-ReID feature extraction with a verified pretrained checkpoint."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor

LOGGER = logging.getLogger("person_reid")

CHECKPOINT_NAME = "osnet_ain_x1_0_msmt17.pth"
CHECKPOINT_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_"
    "b64_fb10_softmax_labsmth_flip_jitter.pth"
)
CHECKPOINT_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"


def prepare_rgb_crop(crop: np.ndarray) -> np.ndarray:
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
        raise ValueError("Each person crop must be a non-empty 3-channel image.")
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def ensure_checkpoint(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = weights_dir / CHECKPOINT_NAME

    if checkpoint.exists():
        actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_hash != CHECKPOINT_SHA256:
            raise ValueError(
                f"OSNet checkpoint checksum mismatch for {checkpoint}; "
                "remove the invalid file and run again to download a verified copy."
            )
        return checkpoint

    temporary = checkpoint.with_suffix(checkpoint.suffix + ".download")
    LOGGER.info("Downloading OSNet person-ReID weights from Hugging Face")
    try:
        request = urllib.request.Request(
            CHECKPOINT_URL, headers={"User-Agent": "person-reid-course-project"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Downloaded OSNet checkpoint failed SHA-256 verification.")
        temporary.replace(checkpoint)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return checkpoint


class OSNetEmbedder:
    """Extract L2-normalized 512-dimensional person-ReID embeddings."""

    def __init__(self, device: torch.device, weights_dir: Path = Path("weights")):
        checkpoint = ensure_checkpoint(weights_dir)
        LOGGER.info("Loading MSMT17-trained OSNet person-ReID embeddings on %s", device)
        self.extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=str(checkpoint),
            device=str(device),
            verbose=False,
        )

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        rgb_crops = [prepare_rgb_crop(crop) for crop in crops]
        features = self.extractor(rgb_crops)
        if features.ndim != 2 or features.shape != (len(crops), 512):
            raise RuntimeError(
                f"Expected OSNet embeddings with shape ({len(crops)}, 512), "
                f"got {tuple(features.shape)}."
            )
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)
