"""Detect people in a video and match their appearance to a labeled image gallery."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from embedder import OSNetEmbedder
from face_embedder import FaceEmbedder
from tracker import AppearanceTracker

LOGGER = logging.getLogger("person_reid")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match people detected in a video against a folder of labeled person crops."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video file.")
    parser.add_argument(
        "--gallery",
        required=True,
        type=Path,
        help="Gallery root; each immediate subfolder is one identity and contains cropped person images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reidentified.mp4"),
        help="Annotated output video path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/detections.csv"),
        help="Per-detection results CSV path.",
    )
    parser.add_argument("--detector-weights", default="yolov8n.pt")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Detector inference size in pixels."
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum missed frames for a track to remain eligible for re-association.",
    )
    parser.add_argument(
        "--track-similarity-threshold",
        type=float,
        default=0.55,
        help="Minimum cosine similarity used when associating detections with tracks.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.75,
        help="Minimum OSNet cosine similarity to accept a body-only gallery match.",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.70,
        help="Minimum FaceNet cosine similarity for a face-assisted match; calibrate on validation videos.",
    )
    parser.add_argument(
        "--face-body-min",
        type=float,
        default=0.35,
        help="Minimum OSNet similarity required alongside a face match.",
    )
    parser.add_argument(
        "--face-top-k",
        type=int,
        default=5,
        help="Maximum high-scoring person detections per frame to check for a face.",
    )
    parser.add_argument(
        "--face-check-interval",
        type=int,
        default=10,
        help="Minimum frames between face checks for the same track.",
    )
    parser.add_argument("--device", default="auto", help="Embedding device: auto, cpu, cuda, or mps.")
    parser.add_argument("--show", action="store_true", help="Display the annotated video while processing.")
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-enabled PyTorch device is available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available on this machine.")
    return torch.device(requested)


def load_gallery(
    gallery_dir: Path, embedder: OSNetEmbedder, face_embedder: FaceEmbedder
) -> tuple[list[str], np.ndarray, list[np.ndarray | None]]:
    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory does not exist: {gallery_dir}")

    identities: list[str] = []
    templates: list[np.ndarray] = []
    face_templates: list[np.ndarray | None] = []
    for identity_dir in sorted(path for path in gallery_dir.iterdir() if path.is_dir()):
        paths = sorted(
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            continue

        crops = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read gallery image: {path}")
            crops.append(image)

        features = embedder.encode(crops)
        template = features.mean(axis=0)
        template /= max(float(np.linalg.norm(template)), 1e-12)
        identities.append(identity_dir.name)
        templates.append(template)

        face_indices, face_features = face_embedder.encode(crops)
        if len(face_indices):
            face_template = face_features.mean(axis=0)
            face_template /= max(float(np.linalg.norm(face_template)), 1e-12)
            face_templates.append(face_template)
        else:
            face_templates.append(None)
            LOGGER.warning(
                "No face detected in gallery images for %s; body-only matching remains available",
                identity_dir.name,
            )

    if not identities:
        raise ValueError(
            f"No gallery images found. Add cropped .jpg/.png images under identity folders in {gallery_dir}."
        )
    return identities, np.stack(templates), face_templates


def crop_box(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def classify_identity(
    identities: list[str],
    body_scores: np.ndarray,
    face_match: tuple[int, float] | None,
    body_threshold: float,
    face_threshold: float,
    face_body_min: float,
) -> tuple[str, str]:
    body_index = int(np.argmax(body_scores))
    body_similarity = float(body_scores[body_index])
    if (
        face_match is not None
        and face_match[0] == body_index
        and face_match[1] >= face_threshold
        and body_similarity >= face_body_min
    ):
        return identities[body_index], "face+body"
    if body_similarity >= body_threshold:
        return identities[body_index], "body"
    return "Unknown", "unknown"


def persist_verified_track_identity(
    track_id: int,
    identity: str,
    match_method: str,
    verified_tracks: dict[int, str],
) -> tuple[str, str]:
    if match_method == "face+body":
        verified_tracks[track_id] = identity
    elif track_id in verified_tracks:
        return verified_tracks[track_id], "track-confirmed"
    return identity, match_method


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not -1.0 <= args.match_threshold <= 1.0:
        raise ValueError("--match-threshold must be between -1 and 1.")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.max_age < 0:
        raise ValueError("--max-age must be non-negative.")
    if not -1.0 <= args.track_similarity_threshold <= 1.0:
        raise ValueError("--track-similarity-threshold must be between -1 and 1.")
    if not 0.0 <= args.face_threshold <= 1.0:
        raise ValueError("--face-threshold must be between 0 and 1.")
    if not 0.0 <= args.face_body_min <= 1.0:
        raise ValueError("--face-body-min must be between 0 and 1.")
    if args.face_top_k < 1:
        raise ValueError("--face-top-k must be positive.")
    if args.face_check_interval < 1:
        raise ValueError("--face-check-interval must be positive.")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32 pixels.")
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.video}")

    device = get_device(args.device)
    embedder = OSNetEmbedder(device)
    face_embedder = FaceEmbedder(device)
    identities, gallery_features, gallery_face_features = load_gallery(
        args.gallery, embedder, face_embedder
    )
    LOGGER.info("Loaded %d identities from the gallery", len(identities))
    tracker = AppearanceTracker(
        max_age=args.max_age,
        min_cosine_similarity=args.track_similarity_threshold,
    )

    LOGGER.info("Loading person detector weights: %s", args.detector_weights)
    detector = YOLO(args.detector_weights)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The input video reports invalid frame dimensions.")
    if not np.isfinite(fps) or fps <= 0:
        LOGGER.warning("Input FPS is unavailable; using 25 FPS for the output.")
        fps = 25.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {args.output}")

    frame_index = 0
    detection_count = 0
    face_checked_at: dict[int, int] = {}
    verified_tracks: dict[int, str] = {}
    start_time = time.perf_counter()
    try:
        with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
            results_writer = csv.writer(csv_file)
            results_writer.writerow(
                [
                    "frame",
                    "time_seconds",
                    "detection",
                    "track_id",
                    "identity",
                    "body_similarity",
                    "face_similarity",
                    "match_method",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            )

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                result = detector.predict(
                    source=frame,
                    classes=[0],
                    conf=args.confidence,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                else:
                    boxes = np.empty((0, 4), dtype=np.float32)

                crops = [crop_box(frame, box) for box in boxes]
                valid = [(box, crop) for box, crop in zip(boxes, crops) if crop is not None]
                valid_boxes = np.asarray(
                    [box for box, _ in valid], dtype=np.float32
                ).reshape(-1, 4)
                features = embedder.encode([crop for _, crop in valid])
                assignments = tracker.update(valid_boxes, features)
                detection_count += len(valid)
                track_ids = {
                    assignment.detection_index: assignment.track_id
                    for assignment in assignments
                }
                if valid:
                    similarities = features @ gallery_features.T
                    best_body_scores = similarities.max(axis=1)
                    frame_face_matches: dict[int, tuple[int, float]] = {}
                    face_candidates = [
                        int(index)
                        for index in np.argsort(best_body_scores)[::-1]
                        if best_body_scores[index] >= args.face_body_min
                        and frame_index
                        - face_checked_at.get(track_ids[int(index)], -args.face_check_interval)
                        >= args.face_check_interval
                    ][: args.face_top_k]
                    if face_candidates:
                        for index in face_candidates:
                            face_checked_at[track_ids[index]] = frame_index
                        face_indices, face_features = face_embedder.encode(
                            [valid[index][1] for index in face_candidates]
                        )
                        available_gallery_indices = [
                            index
                            for index, template in enumerate(gallery_face_features)
                            if template is not None
                        ]
                        if available_gallery_indices:
                            face_gallery = np.stack(
                                [
                                    gallery_face_features[index]
                                    for index in available_gallery_indices
                                ]
                            )
                            for face_index, face_feature in zip(face_indices, face_features):
                                detection_index = face_candidates[int(face_index)]
                                face_scores = face_feature @ face_gallery.T
                                best_face_index = int(np.argmax(face_scores))
                                identity_index = available_gallery_indices[best_face_index]
                                face_score = float(face_scores[best_face_index])
                                frame_face_matches[detection_index] = (
                                    identity_index,
                                    face_score,
                                )

                    for detection_index, ((box, _), scores) in enumerate(zip(valid, similarities)):
                        best_index = int(np.argmax(scores))
                        body_similarity = float(scores[best_index])
                        track_id = track_ids[detection_index]
                        face_match = frame_face_matches.get(detection_index)
                        face_similarity = face_match[1] if face_match else float("nan")
                        identity, match_method = classify_identity(
                            identities,
                            scores,
                            face_match,
                            args.match_threshold,
                            args.face_threshold,
                            args.face_body_min,
                        )
                        identity, match_method = persist_verified_track_identity(
                            track_id,
                            identity,
                            match_method,
                            verified_tracks,
                        )
                        x1, y1, x2, y2 = (int(value) for value in box)
                        if match_method in {"face+body", "track-confirmed"}:
                            color = (0, 200, 0)
                        elif match_method == "body":
                            color = (255, 180, 0)
                        else:
                            color = (0, 165, 255)
                        label = f"ID {track_id}: {identity} (body {body_similarity:.2f}"
                        if np.isfinite(face_similarity):
                            label += f", face {face_similarity:.2f}"
                        elif match_method == "track-confirmed":
                            label += ", confirmed track"
                        label += ")"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        results_writer.writerow(
                            [
                                frame_index,
                                f"{frame_index / fps:.3f}",
                                detection_index,
                                track_id,
                                identity,
                                f"{body_similarity:.4f}",
                                f"{face_similarity:.4f}" if np.isfinite(face_similarity) else "",
                                match_method,
                                x1,
                                y1,
                                x2,
                                y2,
                            ]
                        )

                writer.write(frame)
                if args.show:
                    cv2.imshow("Person re-identification", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        LOGGER.info("Stopped by user.")
                        break
                frame_index += 1
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    LOGGER.info("Processed %d frames", frame_index)
    elapsed = time.perf_counter() - start_time
    if elapsed > 0:
        LOGGER.info(
            "Processed %d person detections in %.2f seconds (%.2f FPS)",
            detection_count,
            elapsed,
            frame_index / elapsed,
        )
    LOGGER.info("Annotated video: %s", args.output)
    LOGGER.info("Detection results: %s", args.csv)


if __name__ == "__main__":
    main()
