# PathoROB Benchmark

PathoROB evaluates how **invariant** learned representations are to **domain shifts** (different medical centers/scanners). It runs three complementary evaluations.

---

## 1. RI (Robustness Index)

**Task**: Measure if embeddings cluster by biological class rather than by scanner/center.

**How it works**:
- Constructs valid 2x2 pairs: (2 classes x 2 centers)
- For each sample, finds k nearest neighbors in embedding space
- Counts neighbor types:
  - **SO** (Same class, Opposite center) — good
  - **OS** (Opposite class, Same center) — bad

**Metric**: `RI = SO / (SO + OS)`
- **RI -> 1.0**: Representations are center-invariant (neighbors share class, not center)
- **RI -> 0.0**: Representations encode center more than class

---

## 2. APD (Average Performance Drop)

**Task**: Measure robustness to label-center correlation shifts.

**How it works**:
- Creates train splits with varying **correlation levels (rho)** between class labels and centers:
  - rho=0 (base): Balanced (each center has equal class distribution)
  - rho=1: Maximum association (each class dominated by one center)
- Trains a **linear probe** on each split
- Evaluates on:
  - **ID test**: Same centers as training
  - **OOD test**: Held-out centers

**Computation**:

APD measures the **average relative performance drop** across all biased splits (rho > 0) compared to the balanced baseline (rho = 0). When multiple correlation levels are used (e.g., rho = 0.0, 0.5, 1.0), the drops at each biased level are computed and then averaged together.

```
APD = mean(relative_drop for each rho > 0)

where relative_drop = (acc[rho] - acc[base]) / acc[base]
```

For example, with `correlation_levels: [0.0, 0.5, 1.0]`:
| rho | acc_id | relative drop |
|-----|--------|---------------|
| 0.0 | 0.80   | (baseline)    |
| 0.5 | 0.76   | (0.76-0.80)/0.80 = -0.05 |
| 1.0 | 0.72   | (0.72-0.80)/0.80 = -0.10 |

**APD = mean([-0.05, -0.10]) = -0.075**

**Metrics**:
- `apd_id`: Average relative drop on in-distribution across all rho > 0
- `apd_ood`: Average relative drop on out-of-distribution across all rho > 0
- `apd_avg`: Average of apd_id and apd_ood
- `acc_id_rho{X}`: Raw accuracy on ID at correlation level X (for debugging)
- `acc_ood_rho{X}`: Raw accuracy on OOD at correlation level X (for debugging)

**Interpretation**: APD closer to 0 (or positive) = more robust to spurious correlations

### Spurious Correlation Splits

**Why this matters**: In real clinical datasets, class labels are often correlated with medical centers (e.g., a rare disease may only be present at specialized hospitals). Non-robust FMs encode center-specific artifacts that downstream models can exploit as "Clever Hans" shortcuts—achieving high training accuracy via spurious correlations that fail to generalize.

**Cramér's V** measures the association between two categorical variables (here: biological class and medical center). It ranges from 0 (no association) to 1 (perfect association).

**Split construction**: Each split keeps the same total training patches but varies their distribution across the class-center matrix:

```
V = 0.00 (balanced)          V = 1.00 (fully correlated)
         Normal  Tumor                 Normal  Tumor
RUMC  [  2100    2100  ]      RUMC  [    0     4200  ]
UMCU  [  2100    2100  ]      UMCU  [  4200      0   ]
```

At V=0, each center contributes equally to each class. At V=1, each center is associated with exactly one class—a downstream model can "cheat" by predicting class based on center artifacts.

**Intermediate levels** (V = 0.14, 0.29, ..., 0.86) gradually shift from balanced to correlated, revealing how quickly a model's generalization degrades as spurious correlations increase.

**Note**: The PathoROB datasets provide raw patches with class/center labels. The correlation splits are **not pre-computed**—you generate them by sampling patches according to allocation matrices for each V level (see paper Supplementary Note D, Figures 14-16 for exact allocations).

---

## 3. Clustering Score

**Task**: Measure if unsupervised clusters align with biology, not scanner.

**How it works**:
- Runs k-means clustering on embeddings
- Computes Adjusted Rand Index (ARI) against:
  - Biological labels (e.g., normal vs tumor)
  - Center labels (e.g., RUMC, UMCU, etc.)

**Metric**: `score = ARI(clusters, bio_labels) - ARI(clusters, center_labels)`
- **Positive**: Clusters capture biology
- **Negative**: Clusters capture center artifacts
- **Zero**: Random or mixed

---

## Summary Table

| Metric | Question Answered | Good Value |
|--------|-------------------|------------|
| **RI** | Do neighbors share class or center? | -> 1.0 |
| **APD** | Does performance drop with biased training? | -> 0.0 |
| **Clustering** | Do clusters reflect biology or artifacts? | -> positive |

---

## Datasets

PathoROB uses **4 datasets** from 3 public clinical sources, designed to enable comparisons between biological and technical (medical center) signals in FM representations.

### Design Principles

- **Balanced multi-center design**: Equal patches per class-center combination where possible
- **ID/OOD splits**: Both in-distribution and out-of-distribution centers for generalization testing
- **Slide-level case IDs**: Enables proper train/test splitting without data leakage

### Summary

| Dataset | Task | Classes | Centers | Patches | License |
|---------|------|---------|---------|---------|---------|
| Camelyon | Tumor detection | 2 | 5 (2 ID + 3 OOD) | 22,402 | CC0 1.0 |
| TCGA 4x4 | Cancer type | 4 | 8 (4 ID + 4 OOD) | 8,160 | CC-BY-NC-SA 4.0 |
| TCGA 2x2 | Cancer type pairs | varies | varies | 112,800 | CC-BY-NC-SA 4.0 |
| Tolkach ESCA | Tissue compartment | 6 | 4 (3 ID + 1 OOD) | 16,300 | Non-commercial |

**Total**: ~99k patches, 28 biological classes, 34 medical centers

---

### 1. Camelyon

Lymph node metastasis detection from the CAMELYON16/17 challenges.

| Property | Value |
|----------|-------|
| **Task** | Binary tumor detection in breast cancer sentinel lymph nodes |
| **Classes** | `normal`, `tumor` |
| **Patch size** | 256×256 px @ 20x magnification (0.5 µm/pixel) |

**Medical Centers:**

| Center | Role | Patches | Slides |
|--------|------|---------|--------|
| RUMC | ID | 10,200 | 34 |
| UMCU | ID | 10,200 | 34 |
| CWZ | OOD | 662 | 10 |
| RST | OOD | 670 | 11 |
| LPON | OOD | 670 | 8 |

**Manifest columns:**
- `label`: `"normal"` or `"tumor"`
- `medical_center`: `"RUMC"`, `"UMCU"`, `"CWZ"`, `"RST"`, or `"LPON"`

**Source:** [CAMELYON17 Grand Challenge](https://camelyon17.grand-challenge.org/Data/) (CC0 1.0 - Public Domain)

---

### 2. TCGA

Pan-cancer tumor type classification from The Cancer Genome Atlas, via the TCGA-UT dataset.

| Property | Value |
|----------|-------|
| **Task** | Cancer type prediction from tumor patches |
| **Patch size** | 256×256 px @ 20x magnification (0.5 µm/pixel) |

**Biological Classes:**

| Code | Cancer Type |
|------|-------------|
| BRCA | Breast invasive carcinoma |
| COAD | Colon adenocarcinoma |
| LUAD | Lung adenocarcinoma |
| LUSC | Lung squamous cell carcinoma |

#### TCGA 4x4 Configuration

Used for APD (Average Performance Drop) evaluation.

**Medical Centers:**

| Center | Role |
|--------|------|
| Asterand, Christiana, Roswell Park, Uni Pittsburgh | ID |
| Cureline, IGC, Greater Poland, Johns Hopkins | OOD |

- 360 patches per class-center combination (ID)
- 8,160 total patches

#### TCGA 2x2 Configuration

Used for Robustness Index (RI) calculation. Contains 94 unique 2×2 combinations (2 cancer types × 2 medical centers each).

- 300 patches per class-center cell
- 1,200 patches per combination
- 112,800 total patches across all 94 combinations

**Manifest columns:**
- `label`: `"BRCA"`, `"COAD"`, `"LUAD"`, or `"LUSC"`
- `medical_center`: Institution name (e.g., `"Asterand"`, `"Christiana"`)

**Source:** [Zenodo (TCGA-UT)](https://zenodo.org/records/5889558) (CC-BY-NC-SA 4.0)

---

### 3. Tolkach ESCA

Tissue compartment classification in surgically resected oesophageal adenocarcinoma specimens.

| Property | Value |
|----------|-------|
| **Task** | Tissue type classification in oesophageal resections |
| **Patch size** | 256×256 px @ ~0.75 µm/pixel |

**Biological Classes:**

| Code | Tissue Type |
|------|-------------|
| TUMOR | Vital tumor tissue |
| REGR_TU | Regression areas |
| SH_OES | Oesophageal mucosa |
| SH_MAG | Gastric mucosa |
| MUSC_PROP | Muscularis propria |
| ADVENT | Adventitial tissue |

**Medical Centers:**

| Center | Full Name | Role | Patches |
|--------|-----------|------|---------|
| UKK | University Hospital Cologne | ID | 3,000 |
| WNS | Landesklinikum Wiener Neustadt | ID | 5,400 |
| CHA | Charité Berlin | ID | 5,400 |
| TCGA | Multiple TCGA institutions | OOD | 2,500 |

**Manifest columns:**
- `label`: `"TUMOR"`, `"REGR_TU"`, `"SH_OES"`, `"SH_MAG"`, `"MUSC_PROP"`, or `"ADVENT"`
- `medical_center`: `"UKK"`, `"WNS"`, `"CHA"`, or `"TCGA"`

**Source:** [Zenodo (Tolkach ESCA)](https://zenodo.org/records/7548828) (Non-commercial use)

---

### Data Access

All PathoROB datasets are available via HuggingFace:

| Dataset | HuggingFace | Original Source |
|---------|-------------|-----------------|
| Camelyon | [bifold-pathomics/PathoROB-camelyon](https://huggingface.co/datasets/bifold-pathomics/PathoROB-camelyon) | [Grand Challenge](https://camelyon17.grand-challenge.org/Data/) |
| TCGA | [bifold-pathomics/PathoROB-tcga](https://huggingface.co/datasets/bifold-pathomics/PathoROB-tcga) | [Zenodo](https://zenodo.org/records/5889558) |
| Tolkach ESCA | [bifold-pathomics/PathoROB-tolkach_esca](https://huggingface.co/datasets/bifold-pathomics/PathoROB-tolkach_esca) | [Zenodo](https://zenodo.org/records/7548828) |

**Collection:** [huggingface.co/collections/bifold-pathomics/pathorob](https://huggingface.co/collections/bifold-pathomics/pathorob-6899f50a714f446d0c974f87)

---

## Configuration Example

```yaml
tuning:
  plugins:
    - type: "pathorob"
      enable: true
      tune_every: 5
      max_pairs: 0  # 0 = use all valid 2x2 pairs

      ri:
        enable: true
        k_selection_policy: "paper_median_fixed"
        default_k: 21

      apd:
        enable: true
        repetitions: 20
        id_test_fraction: 0.2
        correlation_levels: [0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0]

      clustering:
        enable: true
        repeats: 50
        k_min: 2
        k_max: 30

      datasets:
        camelyon:
          enable: true
          manifest_csv: "/path/to/benchmark.csv"
          id_centers: ["RUMC", "UMCU"]      # In-distribution centers
          ood_centers: ["CWZ", "RST", "LPON"]  # Out-of-distribution centers
```

---

## Dataset Manifest Format

CSV with required columns:

| Column | Description |
|--------|-------------|
| `image_path` | Path to image file |
| `label` | Class label (e.g., "normal", "tumor") |
| `medical_center` | Scanner/center identifier |
| `slide_id` | Slide identifier (for train/test splitting) |

Optional: `sample_id` (auto-generated if missing)
