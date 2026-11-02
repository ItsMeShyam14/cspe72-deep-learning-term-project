# Person Re-identification in a Video

> **Submission draft.** Replace every bracketed field and add results/screenshots from an actual run. Do not submit unmeasured results as facts.

## Team details

| Name | Roll number | Email | Contact number | Individual contribution |
|---|---|---|---|---|
| Shyam Sunder S | 106123123 | [Not provided] | [Not provided] | [To be completed by the team] |
| Abhinandan Reddy J | 106123003 | [Not provided] | [Not provided] | [To be completed by the team] |
| Joshua Joy | 106122061 | [Not provided] | [Not provided] | [To be completed by the team] |

## Problem statement

Given a video and a gallery of labeled person images, detect people in each frame and determine whether each detected person matches one of the gallery identities. If the best match is below a validation-selected threshold, report `Unknown`. The system produces an annotated video and a CSV of per-frame detections and matches.

## Gaps identified

- Many basic video demos detect people but do not compare them with a labeled identity gallery.
- A closed-set matcher always forces a known identity, even when the person is not in its gallery.
- Camera viewpoint, lighting, occlusion, and clothing changes make appearance-based matching unreliable.
- A useful demo should expose scores and bounding boxes and support held-out evaluation instead of relying only on visual impressions.

## Proposed architecture

```text
Input video
    |
    v
YOLOv8-n person detector ------> person bounding boxes
                                      |
                                      v
                              crop each detection
                         /                     \
                        v                       v
          MSMT17-trained OSNet             MTCNN face detection
          512-D body embeddings                 |
                        |                       v
                        |           VGGFace2-trained FaceNet
                        |            512-D face embeddings
                        \                       /
                         \                     /
Labeled gallery -> body and face templates per identity
                                      |
                                      v
                         cosine-similarity comparison
                                      |
                  body/face agreement or Unknown rejection
                                      |
                                      v
              appearance + IoU Hungarian-assignment tracker
                     track IDs through short gaps
                                      |
                     annotated MP4 + detection-results CSV
```

## Algorithmic steps

1. Load the pretrained YOLOv8-n person detector, MSMT17-trained OSNet body embedder, VGGFace2-trained FaceNet face embedder, and labeled gallery images.
2. Compute normalized body embeddings for gallery crops and face embeddings where MTCNN finds a face. Average and normalize templates per identity and modality.
3. Read the input video one frame at a time and detect people.
4. Crop each valid person bounding box and compute its OSNet body embedding.
5. Check up to five high body-similarity detections per frame for faces, retrying each track every ten frames. A face-assisted match requires face/body identity agreement and both modality thresholds; otherwise, only a high-confidence body-only match is accepted, or the detection stays `Unknown`.
6. Match current detections to active tracks using a cost combining cosine distance and bounding-box IoU; use Hungarian assignment so a detection is not assigned to multiple tracks.
7. Age unmatched tracks, retain them for a configurable number of missed frames, and smooth matched appearance features.
8. Draw the track ID, predicted gallery label, body similarity, and available face similarity on the frame; write each detection to CSV.
9. Save the annotated video and report the validation-selected thresholds and held-out evaluation metrics.

## Dataset used

**Demo footage:** The user-provided `Assignment/slides/Recording 2026-10-26 223934.mp4` is an approximately 19-second, 1910 × 826, 30 FPS clip. Inspected frames show a crowded street scene with multiple moving people and occasional occlusion. It is used as a qualitative demo clip, not a labeled benchmark sequence.

**Reference gallery:** The user-provided `Assignment/slides/Screenshot 2026-10-26 224529.png` is stored in `data/gallery/person_01/reference.png` and used as the single reference image for gallery label `person_01`. No ground-truth identity annotations were provided for the video; therefore ReID rank-1, mAP, IDF1, and false-match accuracy cannot be reported from this demo alone. Record the original footage/image sources and any applicable usage terms in the final submission.

**Optional benchmark:** MARS is a video-based person ReID dataset with identity labels and tracklets, so it can support repeatable rank-1/mAP evaluation on a standard dataset. It is not required for this assignment. If using MARS, follow its official terms and document the version, split, preprocessing, and source. This project does not download or contain MARS.

## Tools used

- Python 3.12.6; project virtual environment uses PyTorch 2.14.1+cpu and Torchvision 0.29.1+cpu. Installing matching CUDA-enabled wheels was attempted, but the 2 GB download had repeated connection resets and ended with a Windows file-lock error; the demo therefore ran on CPU.
- Ultralytics 8.4.190; OpenCV 5.0.0; SciPy 1.18.1; NumPy 2.5.4; FaceNet-PyTorch 2.5.3.
- Hardware observed: Intel Core i5-12450H CPU and NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB reported VRAM, driver 581.83.
- AI tools: [Tool name and specific purpose; accurately describe assistance with code, debugging, explanations, or report editing.]

## Results and analysis

The corrected OSNet/FaceNet pipeline processed a 119-frame excerpt from source times 15–18.97 seconds on CPU. It recorded 1,509 person detections. Two frames passed fresh face/body verification; the confirmed target track was outlined in green for 19 consecutive detections, while the other 1,490 detections remained `Unknown`. The fresh-match body/face similarities were `0.393` / `0.805` and `0.358` / `0.760`. After the direct face/body check verifies a track, detections assigned to that same track remain green and are marked `track-confirmed`. The earlier implementation carried face scores across track-ID switches and mislabeled a different person; the current implementation does not carry face scores, and requires a fresh face check for initial verification. A higher-resolution detector probe illustrates why body matching alone is risky: at 15 seconds, a partial yellow sleeve scored `0.617`, while a full-person box of the apparent target scored `0.497`.

The target is outlined in green for the lifetime of its confirmed track; unconfirmed detections remain yellow/orange:

![Target identity highlighted with a green box](./outputs/target_match_excerpt_preview.png)

| Probe | Observed result |
|---|---:|
| OSNet full-person candidate, 15 s | 0.497 body cosine similarity |
| OSNet partial-sleeve crop, 15 s | 0.617 body cosine similarity |
| FaceNet apparent-target face, accepted excerpt frame 75 / source 17.5 s | 0.805 face cosine similarity |
| Configured face-assisted threshold | 0.70 (exploratory; not calibrated) |
| Configured minimum body similarity for face-assisted match | 0.35 (exploratory; not calibrated) |
| Freshly verified face/body detections | 2 |
| Consecutive green detections on verified track | 19 (frames 75–93) |
| Other detections left as `Unknown` | 1,490 |
| Excerpt processing speed | 0.42 FPS on CPU |

These are a small, visually inspected sample—not a labeled validation set, an accuracy estimate, or evidence of reliable identity matching across the full clip. The revised pipeline leaves low-confidence/occluded detections as `Unknown`; it maintains a verified label only while the tracker continues to assign that track ID. Track switches remain possible. The clip has no ground-truth identity or box annotations, so rank-1, mAP, precision/recall, IDF1, and false-match rates cannot be calculated.

An earlier ResNet-only baseline was run end-to-end on CPU: 569 frames, 5,414 detections, 47 numeric tracks, 1.42 FPS, and no accepted match at its default `0.75` threshold (maximum cosine similarity `0.6904`). Those figures describe the old ResNet pipeline, not the revised OSNet/FaceNet implementation.

**Verification:** The corrected 119-frame run completed in 282.86 seconds and produced a 119-frame, 1910 × 826, 30 FPS MPEG-4 excerpt and CSV. Thirteen unit tests passed; Python compilation and `pip check` passed. CUDA inference was not available in the checked environment.

| Metric | Validation | Held-out test |
|---|---:|---:|
| Person detection precision / recall | Not measured (no box annotations) | Not measured (no box annotations) |
| ReID rank-1 accuracy | Not measured (no identity ground truth) | Not measured (no identity ground truth) |
| ReID mAP | Not measured (no identity ground truth) | Not measured (no identity ground truth) |
| Unknown-person false-match rate | Not measurable from this clip | Not measurable from this clip |
| Processing speed (frames/second) | 0.42 FPS on CPU for the 119-frame excerpt | Not measured |

The revised system uses MSMT17-trained OSNet for body ReID and VGGFace2-trained FaceNet for visible-face comparison. The appearance/IoU tracker remains a prototype for maintaining IDs through short gaps in one video; it does not establish cross-camera identity. Sparse frame inspection suggests face embeddings may help in this clip, but the thresholds have not been validated on representative data and the clip has no annotations. Similarity is not proof of identity. Discuss false detections and missed matches, especially under occlusion, blur, poor lighting, similar clothing, and viewpoint changes.

## Analysis and difference from existing approaches

The prototype combines an off-the-shelf person detector, identity-trained body and face embeddings, explicit unknown rejection, and short-term appearance-aware association. It is simpler to run than a ReID training pipeline and provides traceable per-detection outputs. This is an integration contribution, not a novel model; generalization and accuracy remain unmeasured.

## Novel contribution

The project contribution is a reproducible end-to-end demo that joins person detection, OSNet body templates, optional face-assisted verification, cosine-similarity matching, open-set `Unknown` rejection, and CSV/video outputs. This is an integration contribution, not a claim of a new ReID architecture. [Add any original experiment or extension actually completed by the team.]

## Individual contribution

- [Member name]: [specific implementation, experiments, evaluation, or documentation completed]
- [Member name]: [specific contribution]

## Learning outcome

- Understood the difference between person detection, tracking, and person re-identification.
- Implemented body/face feature extraction and cosine-similarity matching against labeled identity templates.
- Observed the need for open-set rejection, threshold validation, and held-out testing.
- Identified limitations due to domain shift, occlusion, viewpoint, and generic pretrained features.

## AI tool usage

| Tool | Purpose | Work reviewed/validated by the team |
|---|---|---|
| Copilot SDK assistant in VS Code | Assisted with the code prototype and tracker, debugging the environment, running tests and the video demo, and drafting documentation. | The tracker tests and full CPU demo were executed; the team should review and understand the implementation and verify source attribution before submission. |

## Code link

[Add the GitHub repository URL after publishing the project.]
