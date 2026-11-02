# Person Re-identification in a Video

This demo detects people in a video, compares detected person crops with a labeled gallery using OSNet body embeddings, and uses local FaceNet face embeddings as an additional match signal when a face is visible. It assigns temporal track IDs using appearance and bounding-box overlap and writes an annotated MP4 plus a CSV with separate body and face similarities.

## Approach

1. YOLOv8-n detects the person class in each video frame.
2. MSMT17-trained OSNet produces a 512-dimensional body-ReID embedding for each person crop.
3. MTCNN detects and aligns faces in the highest body-similarity candidates; VGGFace2-trained FaceNet produces normalized 512-dimensional face embeddings.
4. Cosine similarity compares body and face embeddings to the labeled gallery. A face-assisted match requires the face and body branches to agree on the identity and meet their configured thresholds. High-confidence body-only matches can also be accepted; other detections remain `Unknown`.
5. A Hungarian-assignment tracker combines body-embedding similarity and bounding-box IoU to maintain IDs through short detection gaps.

This is an educational prototype, not a production identity system. Track IDs are short-term, single-camera associations; they do not demonstrate cross-camera or long-term ReID. The default face threshold was explored on only a few frames of the supplied clip and is not a validated operating point. Calibrate thresholds with representative labeled validation footage. Similarity is not proof of identity.

## Project modules

- `reid_video.py`: command-line interface, gallery matching, video processing, visualization, and CSV export.
- `embedder.py`: MSMT17-trained OSNet person-ReID feature extraction.
- `face_embedder.py`: local MTCNN face detection and VGGFace2-trained FaceNet feature extraction.
- `tracker.py`: Hungarian assignment over appearance similarity and box IoU; controls short-term track age and feature smoothing.
- `tests/`: unit tests for tracker behavior and match rejection/consensus logic.

## Setup

Use Python 3.10 or newer. From this folder, create/activate a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

These pinned PyTorch/TorchVision versions were selected for the checked Python 3.12 environment. The requirements install uses PyPI's CPU wheels by default. For NVIDIA GPU acceleration, optionally install the matching CUDA 13.0 wheels **before** installing the rest of the requirements:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

The CUDA wheel command targets the observed RTX 3050 driver; it downloads a large (about 2 GB) wheel and may fail on a slow or interrupted connection. If it fails, use the CPU installation above and `--device auto`. For a different machine or Python version, use the official PyTorch install selector. The program automatically uses CUDA or Apple MPS when available, otherwise CPU. The first run downloads YOLO, OSNet, MTCNN, and FaceNet weights.

### Environment check performed for this project

- Python 3.12.6; project environment uses PyTorch 2.14.1 and TorchVision 0.29.1.
- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), driver 581.83.
- OpenCV 5.0.0; Ultralytics 8.4.190.
- PyTorch runtime is CPU-only (`2.14.1+cpu`); installing the 2 GB CUDA wheel was attempted but interrupted by connection resets and a Windows file-lock error.
- Tracker unit tests passed. The full pipeline was run on CPU; observed results are recorded below.

## Prepare the gallery

Use **cropped images containing one person** per image. Make one directory per identity:

```text
data/
  gallery/
    person_01/
      front.jpg
      side.jpg
    person_02/
      front.jpg
      side.jpg
  input.mp4
```

Use the same identity names consistently and only include images you are authorized to use. A gallery image should show the full person where possible; avoid group images.
The local gallery, generated outputs, virtual environment, and downloaded detector checkpoint are excluded by `.gitignore`; do not publish identifiable gallery images or private videos.

## Run

```powershell
python .\reid_video.py --video .\data\input.mp4 --gallery .\data\gallery
```

Specify output locations, threshold, detector confidence, or preview mode:

```powershell
python .\reid_video.py `
  --video .\data\input.mp4 `
  --gallery .\data\gallery `
  --output .\outputs\demo.mp4 `
  --csv .\outputs\matches.csv `
  --match-threshold 0.75 `
  --face-threshold 0.70 `
  --confidence 0.25 `
  --show
```

The first run may take longer while pretrained weights are downloaded. `--show` opens a preview window; press `q` to stop early. The defaults use a `0.75` body-only threshold, a `0.70` face threshold, and a `0.35` minimum body similarity for face-assisted matching. The face threshold is exploratory for the supplied clip, not validated. Face-assisted labels are emitted only when a face was checked in that frame; they are not propagated across frames because track IDs can switch. Choose thresholds on a representative labeled validation set and report validation results. A high score is not proof of identity.

The face-assisted path checks up to five high body-similarity detections per frame and retries each track every ten frames to limit CPU cost. Change `--face-top-k` and `--face-check-interval` if needed. The default detector input is 640 pixels; larger inputs can help with small people but increase runtime.

Tune the short-term association window with `--max-age` (missed frames) and the appearance gate with `--track-similarity-threshold`. The tracker also uses bounding-box IoU. These are prototype defaults, not validated universal thresholds.

### Run the supplied demo clip

From this project folder, the provided clip and gallery can be run with:

```powershell
.\.venv\Scripts\python.exe .\reid_video.py `
  --video "..\slides\Recording 2026-10-26 223934.mp4" `
  --gallery .\data\gallery `
  --output .\outputs\slides_demo.mp4 `
  --csv .\outputs\slides_demo.csv `
  --device auto
```

The video is crowd-heavy, so missed detections and ID switches are possible. The single reference image does not provide ground-truth labels for the entire clip; treat the output as a qualitative demonstration.

### Observed demo run

The earlier ResNet-only baseline completed the supplied clip on CPU: 569 frames, 5,414 person detections, 47 numeric track IDs, and approximately 1.42 processed frames/second. At its `0.75` threshold, all detections were `Unknown`, with a maximum similarity of `0.6904`. This baseline result does not describe the current OSNet/FaceNet code and is not an identity-match result.

### Corrected OSNet/FaceNet excerpt

The corrected pipeline processed a 119-frame excerpt from source times 15–18.97 seconds on CPU. It recorded 1,509 detections. Two frames passed fresh face/body verification; the verified target track was green in 19 consecutive detections, while the other 1,490 detections remained `Unknown`. The fresh-match body/face similarities were `0.393` / `0.805` and `0.358` / `0.760`. Processing took 282.86 seconds (`0.42` FPS). This is a short qualitative demonstration, not an accuracy evaluation.

After a face and body match verifies an identity for a track, subsequent detections assigned to that same track stay green and are marked `confirmed track`; other tracks remain yellow/orange. In this excerpt the target's verified track remained green from frame 75 through frame 93. A higher-resolution detector probe showed why body matching alone is risky: a partial sleeve crop scored `0.617`, compared with `0.497` for a full-person crop. Track IDs can switch, so persistent labels remain a qualitative aid rather than ground-truth identity.

The excerpt output is [outputs/target_match_excerpt.mp4](./outputs/target_match_excerpt.mp4), its detection CSV is [outputs/target_match_excerpt.csv](./outputs/target_match_excerpt.csv), and a still is [outputs/target_match_excerpt_preview.png](./outputs/target_match_excerpt_preview.png).

## Tests

From this project folder, run the unit tests:

```powershell
python -m pytest .\tests -q
```

Unit tests cover tracker behavior and match decision logic. They do not validate detector quality, face matching accuracy, or ReID accuracy.

## Evaluation

For a defensible report, use labeled clips with identities that appear in the gallery and separate validation/test clips (or camera sessions) to avoid evaluating on the gallery images themselves. Report:

- Detection: person precision/recall or mAP, if ground-truth boxes are available.
- Re-identification: rank-1 accuracy and mAP on labeled query detections.
- Tracking: IDF1, ID switches, and track fragmentation only when ground-truth track annotations are available.
- Open-set behavior: false-match rate for identities absent from the gallery, at the selected threshold.
- Runtime: processed frames per second and hardware used.

Do not claim accuracy from a visual demo or from the similarity score alone. This repository does not include a dataset, video, gallery, or measured results.

## Suggested dataset

MARS is optional, not required. It is a video-based person ReID dataset with labeled tracklets that can support repeatable benchmark evaluation, subject to its official access terms. Market-1501 is an image-based ReID dataset, not a video dataset. For a one-day project demo, use a short, authorized video containing people and prepare a small labeled gallery; clearly identify this as a demonstration rather than a benchmark. The project does not include either dataset.

## Limitations and responsible use

Appearance and face embeddings can fail under viewpoint changes, occlusion, blur, lighting changes, or domain shift. The exploratory threshold is not calibrated, and a similarity score is not proof of identity. Do not use this educational demo for consequential decisions or to match people without authorization. Store video and gallery data securely, limit access and retention, and follow applicable consent and privacy requirements.
