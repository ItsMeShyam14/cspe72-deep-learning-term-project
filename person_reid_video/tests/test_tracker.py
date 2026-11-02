import numpy as np
import pytest

from tracker import AppearanceTracker


def test_assigns_ids_and_keeps_them_for_similar_appearances():
    tracker = AppearanceTracker()
    first = tracker.update(
        np.array([[0, 0, 20, 40]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    second = tracker.update(
        np.array([[2, 0, 22, 40]], dtype=np.float32),
        np.array([[0.99, 0.01]], dtype=np.float32),
    )

    assert first[0].track_id == second[0].track_id == 1


def test_recovers_track_after_short_detection_gap():
    tracker = AppearanceTracker(max_age=2)
    first = tracker.update(
        np.array([[0, 0, 20, 40]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    tracker.update(np.empty((0, 4)), np.empty((0, 2)))
    recovered = tracker.update(
        np.array([[1, 0, 21, 40]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )

    assert recovered[0].track_id == first[0].track_id


def test_does_not_recover_track_after_max_age():
    tracker = AppearanceTracker(max_age=1)
    first = tracker.update(
        np.array([[0, 0, 20, 40]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    tracker.update(np.empty((0, 4)), np.empty((0, 2)))
    tracker.update(np.empty((0, 4)), np.empty((0, 2)))
    expired = tracker.update(
        np.array([[0, 0, 20, 40]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )

    assert expired[0].track_id != first[0].track_id


def test_assigns_different_ids_to_unmatched_appearance():
    tracker = AppearanceTracker(min_iou=0.2)
    first = tracker.update(
        np.array([[0, 0, 20, 40]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    second = tracker.update(
        np.array([[100, 0, 120, 40]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.float32),
    )

    assert second[0].track_id != first[0].track_id


def test_empty_detections_are_supported():
    tracker = AppearanceTracker()

    assert tracker.update(np.empty((0, 4)), np.empty((0, 0))) == []


def test_rejects_invalid_boxes_and_feature_shapes():
    tracker = AppearanceTracker()

    with pytest.raises(ValueError, match="positive width"):
        tracker.update(
            np.array([[0, 0, 0, 10]], dtype=np.float32),
            np.array([[1, 0]], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="matching boxes"):
        tracker.update(
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )


def test_rejects_zero_length_embedding():
    tracker = AppearanceTracker()

    with pytest.raises(ValueError, match="non-zero norm"):
        tracker.update(
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.array([[0, 0]], dtype=np.float32),
        )
