# VolleySight — Quantitative Evaluation & Ablation Results

## 1. Ablation Study #1: Scene-Background Data Leakage (Random vs Video-Grouped Split)

When image tiles (`768 × 768`) from stationary arena cameras (`5.0s` intervals) are split randomly across Train/Val/Test, the detector memorizes static court backgrounds, lighting, and spectator stands from neighboring frames (`t ± 5s`). Furthermore, `69.3%` (`79 / 114`) of the held-out test match (`mak0QV2pk`) tiles appear inside `random_split`'s training set.

| Training Split | Evaluation Split | Scene Overlap with Train | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Split** (`seed=42`) | Random Val (`210` tiles) | **Leaked** (`10/11` videos shared) | 0.914 | 0.918 | **0.944** | **0.608** |
| **Video-Grouped Split** | Group Val (`209` tiles, `2` unseen matches) | **0 shared videos** (`majXaA68z`, `maNen4zOD`) | 0.888 | 0.866 | **0.894** | **0.469** |
| **Random Split** (`seed=42`) | Group Test (`114` tiles, `mak0QV2pk`) | **Leaked** (`79/114` tiles in train) | — | — | **0.970** | **0.678** |
| **Video-Grouped Split** | Group Test (`114` tiles, `mak0QV2pk`) | **0 shared video IDs** | 0.912 | 0.904 | **0.927** | **0.488** |

> **Key Takeaway:** Random image splitting artificially inflates validation `mAP@50` by **`+5.0%`** (`0.894 → 0.944`) and `mAP@50-95` by **`+13.9%`** (`0.469 → 0.608`). Video-grouped splitting is mandatory for honest evaluation on fixed-camera sports datasets.

---

## 2. Per-Class Performance (`YOLO11s`, Video-Grouped Test Set)

| Class ID | Class Name | Test Instances | Box Area (`768×768`) | AP@50 | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | `Player` | 426 | `84 × 180 px` (`2.57%`) | **0.989** | **0.977** |
| `1` | `Referee` | 72 | `90 × 157 px` (`2.40%`) | **0.977** | **0.978** |
| `2` | `Ball` | 45 | `33 × 34 px` (`0.19%`) | **0.817** | **0.756** |
| **All** | **Overall** | **543** | — | **0.927** | **0.904** |

---

## 3. Ablation Study #2a: Small-Object (`Ball`) Multi-Scale & High-Res Inference

Because `Ball` occupies `< 0.2%` of a tile (`~33 × 34 px`), evaluating higher inference resolution and Test-Time Augmentation (TTA) improves `Ball AP@50` by up to **`+4.6%`**:

| Inference Mode | `Ball` AP@50 | `Ball` Recall | Overall mAP@50 |
| :--- | :---: | :---: | :---: |
| Standard Native Resolution (`imgsz=768`) | 0.817 | **0.756** | 0.927 |
| Test-Time Augmentation (`augment=True`) | 0.842 (`+2.5%`) | 0.667 | **0.935** |
| High-Resolution Upscale (`imgsz=1152`) | **0.863 (`+4.6%`)** | 0.644 | 0.931 |

---

## 4. Ablation Study #2b: Cross-Tile Seam Box Merging (`3840 × 768` Panorama)

Across all `309` panoramic timestamps, `66` ground-truth objects are sliced across adjacent vertical tile seams (`x ∈ {768, 1536, 2304, 3072}`). Evaluating panoramic predictions against seam-merged ground truth on the test match (`mak0QV2pk`, `IoU >= 0.5`, `conf >= 0.25`):

| Panoramic Post-Processing | Precision | Recall | False Positives (FP) | FP Reduction |
| :--- | :---: | :---: | :---: | :---: |
| Raw Per-Tile Predictions (Unmerged) | 0.818 | **0.962** | 113 | Baseline |
| **Cross-Tile Seam Merger + Global NMS** (`delta_x=12, IoU_y>=0.3`) | **0.891 (`+7.3%`)** | 0.945 | **61** | **`-46.0%`** |
