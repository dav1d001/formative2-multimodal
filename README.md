# Formative 2 – Multimodal Data Preprocessing
**User Identity & Product Recommendation System** · ALU · 2026 September term

A product-recommendation system that only answers verified users: a **face** is recognised first, a **voice** phrase confirms the transaction, and only then the **product model** predicts what the customer is likely to buy. If either check fails, access is denied.

```
Start ─► Facial Recognition ─► Run Product Recommendation ─► Voice Validation ─► Display Predicted Product
              │ fail                                              │ fail
              ▼                                                   ▼
        Access Denied                                       Access Denied
```

## Project status

| Part | Status |
|---|---|
| Task 1 – Data merge & feature engineering | ✅ Complete |
| Task 2 – Image collection, augmentation, features | ✅ Pipeline complete · ✅ photos for 4 members (Nkuba, David, Rolande, Sheila); add any further members' photos and re-run |
| Task 3 – Audio collection & features | ⏳ In progress (`data/audio/`, `audio_features.csv`) |
| Task 4 – Face, voice and product models | ⏳ In progress (`models/`) |
| Task 6 – CLI system demonstration + video | ⏳ In progress |

## Repository structure

```
├── data/
│   ├── raw/                     provided files: customer_transactions.xlsx, customer_social_profiles.xlsx
│   ├── images/                  original face photos:  <member>_<neutral|smiling|surprised>.jpg
│   ├── images_augmented/        8 augmented copies of every photo (generated)
│   ├── audio/                   voice recordings (Task 3)
│   └── processed/
│       ├── merged_dataset.csv                 all 150 transactions + social features (31 columns)
│       ├── merged_dataset_model_ready.csv     the 117 transactions that have social data
│       └── image_features.csv                 one row per original / augmented image (399 columns)
├── src/
│   ├── merge_pipeline.py        Task 1: clean → aggregate → merge → engineer features
│   ├── image_pipeline.py        Task 2: crop → augment → extract features
│   └── make_demo_images.py      smoke-test helper only (cartoon placeholders, never submitted)
├── notebooks/
│   ├── task1_data_merge.ipynb   cleaning, merge justification, validation, 6 EDA plots
│   └── task2_image_processing.ipynb   photo display, augmentations, feature checks, PCA
├── figures/                     all plots as PNG
├── models/                      trained models (Task 4)
├── report/                      final report PDF
├── report_snippets_tasks_1_2.md text for the report (Tasks 1–2)
└── requirements.txt
```

## Quick start

```bash
git clone <this-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/merge_pipeline.py        # → data/processed/merged_dataset*.csv
python src/image_pipeline.py        # → data/processed/image_features.csv (+ augmented images)
jupyter notebook notebooks/         # open and run the notebooks top to bottom
```
Optional: `python src/image_pipeline.py --embeddings` adds 512-d ResNet-18 face embeddings (needs `pip install torch torchvision`).
Automatic face cropping needs OpenCV **4.x** (pinned in `requirements.txt`); on OpenCV 5 the pipeline falls back to a centre crop and warns.

---

## Task 1 – Data merge & feature engineering

**Inputs:** `customer_transactions` (150 rows, 75 customers, target `product_category`, 5 balanced classes) and `customer_social_profiles` (155 rows, 84 customers).

**Problems found and how they were handled**

| Problem | Solution |
|---|---|
| Customer IDs in different formats (`151` vs `"A151"`) | Normalised to integer `customer_id` |
| 5 exact duplicate social rows | Dropped |
| Several social rows per customer (up to 7) | Aggregated to **one row per customer before joining** |
| 10 missing `customer_rating` (6.7 %) | Median imputation + `rating_missing` flag |
| Only 61 customers appear in both files | **Left join** from transactions + `has_social_profile` flag |

A naive row-level join gives 213 rows (96 phantom rows, 33 real transactions lost). The chosen many-to-one left join (`validate='many_to_one'`) keeps all 150 transactions.

**Engineered features (31 columns):** aggregated social features (mean/max engagement and purchase interest, platform count, dominant platform, sentiment shares and score, engagement × interest); temporal (month, weekday, weekend); amount tier; leakage-safe customer history built only from *earlier* purchases (prior transaction count, prior average amount/rating, days since previous purchase, previous category).

**Validation:** row count preserved, `transaction_id` unique, social features present exactly for matched rows, valid value ranges, hand-recomputed aggregate matches the merged value.

**EDA highlights:** classes balanced (25–35 each); no IQR outliers; engagement and interest roughly uniform; sentiment skews positive (40 / 33 / 27 %); correlations weak (max |r| = 0.43).

**Modelling guidance (important for Task 4):**
* Use **customer-grouped** splits (`GroupKFold` on `customer_id`) – social features repeat for every transaction of a customer.
* Expect **weak signal**: grouped-CV accuracy ≈ 0.20–0.25 vs chance 0.20 and majority baseline 0.205. Report accuracy, macro-F1 and log-loss against that baseline.
* `purchase_amount` and `customer_rating` are only known after a purchase; decide and document whether the model may use them.

## Task 2 – Facial images

* **Photos:** each member provides `<member>_neutral.jpg`, `<member>_smiling.jpg`, `<member>_surprised.jpg`.
* **Preprocessing:** Haar-cascade face detection, 25 % margin crop, resize to 224 × 224.
* **Augmentations (8 per photo):** rotation ±15°, horizontal flip, grayscale, brighter, darker, Gaussian noise, blur.
* **Features:** 16-bin RGB histograms (48), HOG (324), uniform LBP (10), global statistics (≈ 12), optional CNN embedding (512).
* **Output:** `image_features.csv` – 9 rows per photo (1 original + 8 augmented); `image_id` identifies the source photo.
* **Current result:** 12 photos (David, Nkuba, Rolande, Sheila × 3 expressions) → 108 rows × 399 columns (12 originals + 96 augmented); faces detected in 12 / 12.
* **Findings:** expressions do not separate in feature space (silhouette ≈ 0), and identity separation is driven partly by background/lighting – Nkuba (dark outdoor scene) and Rolande separate easily, while David and Sheila (similar plain indoor walls) overlap. For the face model use CNN embeddings (`--embeddings`), tight crops and consistent backgrounds.
* **Leakage rule for Task 4:** split by `image_id`, never by row – a photo's 9 rows are near-duplicates.

## Team contributions

| Member | Contribution |
|---|---|
| Zaga | Task 1 (data merge, cleaning, validation, feature engineering, EDA); Task 2 image pipeline, notebook and `image_features.csv`; repo structure and README |
| Nkuba | Face photos (neutral, smiling, surprised) |
| David | Face photos (neutral, smiling, surprised) |
| Rolande | Face photos (neutral, smiling, surprised) |
| Sheila | Face photos (neutral, smiling, surprised) |

_Update this table as Tasks 3, 4 and 6 are completed – every member's contribution must be listed in the final report._

## Deliverables checklist

- [x] Raw datasets · [x] merged dataset with feature engineering · [x] `image_features.csv`
- [ ] `audio_features.csv` · [ ] face / voice / product model scripts · [ ] command-line app
- [x] Jupyter notebooks (Tasks 1–2) · [ ] report PDF · [ ] demo video link

**Links:** Report: _TBD_ · Demo video: _TBD_

## Privacy
Face and voice recordings are biometric data of real people, shared with each member's consent for this assignment only. Keep this repository private and shared only with the group and instructor.
