"""Local face embeddings used as an auxiliary identity check."""

from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

from embedder import prepare_rgb_crop


class FaceEmbedder:
    """Detect the largest face in each crop and encode it with FaceNet."""

    def __init__(self, device: torch.device):
        self.device = device
        self.detector = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,
            select_largest=True,
            post_process=True,
            device=device,
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def encode(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        face_indices: list[int] = []
        face_tensors: list[torch.Tensor] = []
        for index, crop in enumerate(crops):
            rgb_crop = prepare_rgb_crop(crop)
            face = self.detector(Image.fromarray(rgb_crop))
            if face is not None:
                face_indices.append(index)
                face_tensors.append(face)

        if not face_tensors:
            return np.empty((0,), dtype=np.int64), np.empty((0, 512), dtype=np.float32)

        batch = torch.stack(face_tensors).to(self.device)
        with torch.inference_mode():
            features = self.model(batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return (
            np.asarray(face_indices, dtype=np.int64),
            features.detach().cpu().numpy().astype(np.float32, copy=False),
        )
