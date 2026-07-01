# Deepfake Video Detection

## Why deepfake detection matters

**Deepfakes** are videos in which the face of a person is automatically swapped with the face of
another person, typically using a pretrained **generative adversarial network (GAN)**. As these
models improve, deepfake videos look more and more realistic, and it is becoming increasingly
difficult to tell a genuine video from a manipulated one. The same technology that powers harmless
entertainment can be used for fraud, disinformation, and identity abuse — so a growing body of work
is dedicated to **automatically detecting deepfakes**.

A critical component of any such system is **feature extraction**. A *feature* is a value or set of
values that characterizes and distinctly describes the sample we want to classify — here, a video or
a single video frame. For a given problem it is rarely obvious which features best describe the
signal, but there are principled approaches for finding good descriptors and computing them reliably
for the faces detected in a video. This project builds that feature-extraction foundation step by
step.

As a first hint that the problem is tractable: subtracting a real frame from its corresponding
deepfake frame shows that the manipulation is **spatially localized to the face**, while the
background is untouched — exactly the kind of structured signal a detector can learn.

![Real vs. deepfake frame and their pixel-wise difference](assets/01_real_vs_fake_diff.png)

## The data

We use two paired datasets:

| Dataset | Role | Layout |
|---|---|---|
| **VidTIMIT** | Original / **real** videos | `Data/VidTIMIT/<subject>/<utterance>.avi` |
| **DeepfakeTIMIT** | GAN face-swapped / **fake** videos | `Data/DeepfakeTIMIT/higher_quality/<subject>/<utterance>-video-<source>.avi` |

Real and fake clips correspond by subject and utterance (e.g. real `fadg0/sa1.avi` ↔ fake
`fadg0/sa1-video-fram1.avi`). Each clip is a short talking-head recording at **512×384**, roughly 119
frames. Because each fake is a face swap of a specific real recording, the datasets are naturally
paired for analysis. *(The raw video data is not committed to this repository.)*

## Pipeline so far

The goal of these early milestones is to turn raw videos into a **clean, uniform set of face images**
ready for feature extraction. Each step is a self-contained Jupyter notebook under `Scripts/`.

### 1. Frames → images (`video_frame_to_image.ipynb`)
Recursively enumerate videos with `glob`, decode frames with OpenCV, and inspect their representation
(each frame is an `(H, W, 3)` `uint8` NumPy array, BGR channel order). We save and compare a matching
real/fake frame pair and compute their pixel-wise difference (figure above), confirming edits are
concentrated on the face.

### 2. Face & landmark detection (`face_detection.ipynb`)
Detect faces and facial landmarks in every frame using **MTCNN** (via `facenet-pytorch`). For each
face we extract the bounding box `(x, y, w, h)` and the **left/right eye** coordinates — the inputs
needed to align faces consistently. Detected faces are cropped and cached to disk so detection isn't
repeated.

![MTCNN face box and eye landmarks](assets/02_face_detection.png)

### 3. Alignment & scaling (`face_alignment.ipynb`)
`crop_and_align()` uses the eye locations to **rotate** each face so the eyes are horizontal,
**scale** it so the inter-eye distance is fixed, and **warp** it into a fixed-size canvas
(`256×256`, left eye at 35%). The result: every face is centered the same way and the same size —
essential for downstream comparison.

![Aligned and scaled faces](assets/03_aligned_faces.png)

### 4. Normalization (`face_normalization.ipynb`)
Three normalizations are applied to the aligned faces, targeting the **best representation for a
learning algorithm, not the human eye** (so some outputs no longer look like faces):

- **Global mean/std** — subtract the database mean face and divide by its per-pixel standard
  deviation (output has mean ≈ 0, std = 1).
- **Histogram equalization (CLAHE)** — adaptive, per colour channel, boosting local contrast.
- **Contrast/brightness (MIN-MAX)** — linearly stretch each face to the full 0–255 range.

![Face normalizations](assets/04_normalizations.png)

## Technologies

- **Python 3.10** (conda env `deepfake-detect`)
- **OpenCV** — video decoding, geometric warping (`getRotationMatrix2D`, `warpAffine`), CLAHE, MIN-MAX
- **facenet-pytorch (MTCNN)** on **PyTorch** — face + facial-landmark detection
- **NumPy** — array representation and global statistics
- **Matplotlib** — visualization
- **Jupyter** — one notebook per milestone

## Repository structure

```
Deepfake-Detection/
├── Scripts/
│   ├── video_frame_to_image.ipynb   # Step 1: frames → images, real/fake diff
│   ├── face_detection.ipynb         # Step 2: MTCNN detection + landmarks
│   ├── face_alignment.ipynb         # Step 3: crop_and_align()
│   └── face_normalization.ipynb     # Step 4: three normalizations
├── assets/                          # figures used in this README
├── Data/                            # raw videos + generated crops (not tracked)
└── README.md
```

## Running the notebooks

```bash
conda activate deepfake-detect
jupyter notebook   # then open a notebook in Scripts/ and select the deepfake-detect kernel
```

> On macOS, PyTorch and OpenCV each bundle an OpenMP runtime; the notebooks set
> `KMP_DUPLICATE_LIB_OK=TRUE` in their first cell to avoid a crash when both are imported.

## Status & roadmap

This is the **initial README** covering data preparation (detection → alignment → normalization). It
will be extended as the project grows toward the end goal: **extracting discriminative features from
normalized faces and training a classifier to detect deepfakes.**
