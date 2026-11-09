# Formative 2 – Multimodal Data Preprocessing (Tasks 1 & 2)

```
data/raw/            customer_transactions.xlsx, customer_social_profiles.xlsx   (provided)
data/images/         <member>_<neutral|smiling|surprised>.jpg                    (YOUR photos – add these)
data/images_augmented/   generated: 8 augmented copies per photo
data/processed/      merged_dataset.csv, merged_dataset_model_ready.csv, image_features.csv
src/merge_pipeline.py    Task 1 script      src/image_pipeline.py   Task 2 script
src/make_demo_images.py  smoke-test placeholders only (never submit)
notebooks/task1_data_merge.ipynb (executed)   notebooks/task2_image_processing.ipynb (run after adding photos)
figures/             EDA + image plots (PNG, ready for the report)
```
## Run
```bash
pip install -r requirements.txt
python src/merge_pipeline.py                 # -> data/processed/merged_dataset*.csv
python src/image_pipeline.py                 # -> data/processed/image_features.csv  (needs photos in data/images/)
python src/image_pipeline.py --embeddings    # optional: + 512-d ResNet-18 embeddings
jupyter notebook notebooks/                  # run Task 2 notebook top-to-bottom after adding photos
```
## Photo checklist (per member)
`amara_neutral.jpg`, `amara_smiling.jpg`, `amara_surprised.jpg` – lower-case name, same format for all, one front-facing face, plain background, good light. Get consent; keep the repo private.
