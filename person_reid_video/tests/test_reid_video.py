import numpy as np

from reid_video import classify_identity, persist_verified_track_identity


def test_face_match_requires_body_identity_agreement_and_both_scores():
    identity, method = classify_identity(
        ["person_01", "person_02"],
        np.array([0.48, 0.43]),
        (0, 0.81),
        body_threshold=0.75,
        face_threshold=0.70,
        face_body_min=0.35,
    )

    assert (identity, method) == ("person_01", "face+body")


def test_face_match_is_rejected_when_body_selects_another_identity():
    identity, method = classify_identity(
        ["person_01", "person_02"],
        np.array([0.48, 0.52]),
        (0, 0.90),
        body_threshold=0.75,
        face_threshold=0.70,
        face_body_min=0.35,
    )

    assert (identity, method) == ("Unknown", "unknown")


def test_low_face_score_does_not_override_unknown_body_match():
    identity, method = classify_identity(
        ["person_01"],
        np.array([0.49]),
        (0, 0.69),
        body_threshold=0.75,
        face_threshold=0.70,
        face_body_min=0.35,
    )

    assert (identity, method) == ("Unknown", "unknown")


def test_high_body_similarity_can_match_without_a_face():
    identity, method = classify_identity(
        ["person_01"],
        np.array([0.80]),
        None,
        body_threshold=0.75,
        face_threshold=0.70,
        face_body_min=0.35,
    )

    assert (identity, method) == ("person_01", "body")


def test_verified_identity_stays_assigned_to_the_same_track():
    verified_tracks = {}
    identity, method = persist_verified_track_identity(
        7, "person_01", "face+body", verified_tracks
    )

    assert (identity, method) == ("person_01", "face+body")
    assert persist_verified_track_identity(
        7, "Unknown", "unknown", verified_tracks
    ) == ("person_01", "track-confirmed")


def test_unverified_tracks_are_not_highlighted_as_confirmed():
    assert persist_verified_track_identity(
        8, "Unknown", "unknown", {}
    ) == ("Unknown", "unknown")
