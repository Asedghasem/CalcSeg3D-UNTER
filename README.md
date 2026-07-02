# CalcSeg3D-UNTER
# Deep Learning Workflow for 3D Calcification Segmentation in Cone Beam Computed Tomographic (CBCT) Images

Official implementation of the paper **"Deep Learning Workflow for 3D Calcification Segmentation in Cone Beam Computed Tomographic Images"** (*Informatics in Medicine Unlocked*).

This repository provides a multi-stage workflow for detecting and segmenting **incidental carotid artery calcifications** in 3D CBCT scans. Calcifications of this kind are clinically important (they are strongly associated with cardiovascular disease) but are notoriously hard to segment: they are tiny, sparsely distributed, and visually similar to surrounding anatomy. In our dataset they occupy only about **0.04%** of the total image volume, yet the proposed workflow reaches a **Dice coefficient of 0.64** while remaining lightweight and fast at inference.

---

## Overview

The pipeline combines three ideas:

1. **3D UNETR segmentation.** A vision-transformer-based encoder–decoder (UNETR) captures long-range volumetric dependencies, which is critical for finding small, spatially dispersed calcifications that slice-wise methods miss. The architecture is adapted (fewer layers, tuned hyperparameters) to fit modest hardware.
2. **Volumetric 3D Grad-CAM.** A custom full-volume Grad-CAM extends class activation maps from 2D to the whole 3D volume. Activations from the **third decoder block** were found to align more closely with the true lesion extent than the raw segmentation output.
3. **Grad-CAM-guided post-processing.** The intermediate attention maps are used as a guidance signal in a Euclidean-distance-based region-expansion step that grows anomalous regions **only where the activation map and the model prediction agree**, improving sensitivity while controlling false positives.


---

## Key Results

**Segmentation model comparison** (Dice coefficient):

| Model      | Dice          |
|------------|---------------|
| 3D U-Net   | 0.33 ± 0.08   |
| nnU-Net    | 0.28 ± 0.08   |
| VAE        | 0.49 ± 0.06   |
| **UNETR**  | **0.64 ± 0.013** |

**Ablation study** (contribution of each component):

| Configuration            | Description                                   | Dice |
|--------------------------|-----------------------------------------------|------|
| Baseline UNETR           | No data augmentation                          | 0.59 |
| UNETR + Data Augmentation| Training with augmentation strategies         | 0.63 |
| **Full Framework**       | Pipeline with heatmap-guided refinement       | **0.67** |

The Grad-CAM-guided post-processing improved the Dice coefficient by up to **7%** and recall by up to **14%**, and reduced HD95 (boundary error) in most cases. The largest gains appear in recall, reflecting better detection of small, previously under-segmented lesions.

Evaluation metrics used: **Dice coefficient**, **Recall (sensitivity)**, **Precision**, and the **95th-percentile Hausdorff Distance (HD95)**.

---

## Dataset

The study introduces a novel 3D CBCT dataset of incidental cervical calcifications.

- **Source:** CBCT scans acquired between January 2009 and August 2023, originally for implant surgery and treatment planning.
- **Scanners:** I-CAT, OP300, and Carestream 9600.
- **Size:** 104 scans collected; 12 excluded for documentation errors or cone-beam artifacts → **92 scans** (60 normal, 32 abnormal).
- **Format:** DICOM. Each volume has a matching segmentation mask of identical dimensions.
- **Annotation:** Calcifications were manually segmented in 3D Slicer by an OMR resident and confirmed by a board-certified oral and maxillofacial radiologist.
- **Split:** 17 training / 6 validation / 6 test volumes.

> **Ethics & availability.** The study was approved by the Stony Brook University Institutional Review Board (IRB). Because the data is clinical and patient-derived, access may be restricted. Please see the [Data Availability](#data-availability) section below or contact the corresponding author regarding access.

---

## Method Details

### Model — 3D UNETR
- Transformer-based encoder with a convolutional decoder (U-Net-inspired).
- Input volume: `96 × 96 × 96`, partitioned into non-overlapping `16 × 16 × 16` patches.
- Transformer configuration: **12 layers**, **12 attention heads**, hidden size **768**.
- No dropout (to preserve already-sparse anomaly signal).

### Loss Function
A combination of **Dice loss** (robust to class imbalance) and **cross-entropy loss** (pixel-wise accuracy).

### Data Augmentation
Noise addition, resolution changes, and rotations — chosen to preserve the essential structure of the original scans given only 32 abnormal samples.

### 3D Grad-CAM
Activation maps are computed per slice across the full volume for a target layer (the **third decoder block**), then aggregated into a coherent 3D attention map. See the paper (Algorithm 1) for the full computation.

### Post-Processing (region expansion)
1. Find pixels within a **Euclidean distance** of the initially predicted anomaly pixels.
2. **Binarize** the Grad-CAM map with a threshold.
3. Combine via logical **AND** (distance mask ∧ binarized heatmap), then **OR** with the initial prediction.
4. Selectively expand small candidate regions under dataset-specific constraints (limits on regions per half-slice, cautious expansion of regions under 4 pixels).

**Post-processing hyperparameters:** Euclidean distance threshold = 3, Grad-CAM binarization threshold = 0.7, region expansion = 4 voxels.

### Training Configuration
| Setting        | Value              |
|----------------|--------------------|
| Epochs         | 100                |
| Batch size     | 4                  |
| Optimizer      | Adam               |
| Learning rate  | 1 × 10⁻⁴           |
| GPU            | NVIDIA P100        |
| Input size     | 96 × 96 × 96       |
| Patch size     | 16 × 16 × 16       |

---

## Limitations

- Full-3D processing is memory- and compute-intensive, constraining model complexity.
- The dataset is small and single-center, which may limit generalizability across scanners and populations.
- Ground-truth annotations may carry inter-rater variability (not evaluated here).
- Outputs should always be interpreted alongside the judgment of qualified medical professionals.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{darvishnoori_calcification_segmentation,
  title   = {Deep Learning Workflow for 3D Calcification Segmentation in Cone Beam Computed Tomographic Images},
  author  = {Darvish Noori, Kimia and Bavarnegin, Hamed and Reeves, Kyle and Mirroshandel, Seyed Abolghasem and Mahdian, Mina},
  journal = {Informatics in Medicine Unlocked},
  year    = {2026}
}
```
<!-- Update volume/pages/DOI/year once the paper is published. -->

---

## Authors

- **Kimia Darvish Noori** — University of Guilan
- **Hamed Bavarnegin** — University of Guilan
- **Kyle Reeves, DDS** — Stony Brook University, School of Dental Medicine
- **Seyed Abolghasem Mirroshandel** — University of Guilan *(corresponding author — mirroshandel@guilan.ac.ir)*
- **Mina Mahdian, DDS, MDSc** — Stony Brook University, School of Dental Medicine *(senior author)*

---

## Data Availability

The dataset was collected under IRB approval at Stony Brook University and contains de-identified clinical CBCT scans. Access is subject to restrictions. 


We thank the oral and maxillofacial radiology team at Stony Brook University for data curation and annotation.
