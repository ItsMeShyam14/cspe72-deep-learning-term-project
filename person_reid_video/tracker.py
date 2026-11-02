"""Appearance-aware online association for short gaps in a single video."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    feature: np.ndarray
    missed_frames: int = 0


@dataclass(frozen=True)
class TrackAssignment:
    detection_index: int
    track_id: int


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Appearance features must have non-zero norm.")
    return features / norms


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class AppearanceTracker:
    """Associate detections using appearance and box overlap.

    This tracker can maintain IDs through short detection gaps in a single
    camera view. It is not a cross-camera or long-term identity system.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_cosine_similarity: float = 0.55,
        min_iou: float = 0.10,
        appearance_weight: float = 0.70,
        feature_momentum: float = 0.80,
    ) -> None:
        if max_age < 0:
            raise ValueError("max_age must be non-negative.")
        if not -1.0 <= min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must be between -1 and 1.")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be between 0 and 1.")
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be between 0 and 1.")
        if not 0.0 <= feature_momentum < 1.0:
            raise ValueError("feature_momentum must be in [0, 1).")

        self.max_age = max_age
        self.min_cosine_similarity = min_cosine_similarity
        self.min_iou = min_iou
        self.appearance_weight = appearance_weight
        self.feature_momentum = feature_momentum
        self._tracks: list[Track] = []
        self._next_id = 1
        self._feature_dim: int | None = None

    def update(
        self, boxes: np.ndarray, features: np.ndarray
    ) -> list[TrackAssignment]:
        boxes = np.asarray(boxes, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)

        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        if features.size == 0:
            expected_dim = self._feature_dim or 0
            features = np.empty((0, expected_dim), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4).")
        if features.ndim != 2 or features.shape[0] != boxes.shape[0]:
            raise ValueError("features must have shape (N, D), matching boxes.")
        if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(features)):
            raise ValueError("boxes and features must contain only finite values.")
        if boxes.shape[0] and np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Each box must have positive width and height.")
        if self._feature_dim is not None and features.shape[1] != self._feature_dim:
            raise ValueError(
                f"Expected {self._feature_dim}-dimensional features, got {features.shape[1]}."
            )
        if features.shape[0]:
            features = _normalize_rows(features)
            self._feature_dim = int(features.shape[1])

        for track in self._tracks:
            track.missed_frames += 1
        self._tracks = [
            track for track in self._tracks if track.missed_frames <= self.max_age
        ]

        assignments: list[TrackAssignment] = []
        matched_detections: set[int] = set()

        if self._tracks and boxes.shape[0]:
            similarities = np.stack(
                [features @ track.feature for track in self._tracks], axis=0
            )
            overlaps = np.asarray(
                [
                    [_iou(track.box, box) for box in boxes]
                    for track in self._tracks
                ],
                dtype=np.float32,
            )
            valid = (similarities >= self.min_cosine_similarity) | (
                overlaps >= self.min_iou
            )
            costs = self.appearance_weight * (1.0 - similarities) + (
                1.0 - self.appearance_weight
            ) * (1.0 - overlaps)
            costs[~valid] = 1e6
            track_rows, detection_cols = linear_sum_assignment(costs)

            for track_index, detection_index in zip(track_rows, detection_cols):
                if not valid[track_index, detection_index]:
                    continue
                track = self._tracks[track_index]
                feature = features[detection_index]
                smoothed = (
                    self.feature_momentum * track.feature
                    + (1.0 - self.feature_momentum) * feature
                )
                track.feature = smoothed / max(float(np.linalg.norm(smoothed)), 1e-12                )
                track.box = boxes[detection_index].copy()
                track.missed_frames = 0
                matched_detections.add(int(detection_index))
                assignments.append(
                    TrackAssignment(int(detection_index), track.track_id)
                )

        for detection_index, (box, feature) in enumerate(zip(boxes, features)):
            if detection_index in matched_detections:
                continue
            track = Track(
                track_id=self._next_id,
                box=box.copy(),
                feature=feature.copy(),
            )
            self._next_id += 1
            self._tracks.append(track)
            assignments.append(TrackAssignment(detection_index, track.track_id))

        return sorted(assignments, key=lambda assignment: assignment.detection_index)
