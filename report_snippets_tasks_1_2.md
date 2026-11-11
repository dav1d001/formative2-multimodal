# Report snippets – Tasks 1 & 2 (paste/adapt into your report)

## Task 1 – Data merge
**Datasets.** `customer_transactions` (150 rows, 75 customers, target `product_category`, 5 classes) and `customer_social_profiles` (155 rows, 84 customers: platform, engagement, purchase interest, review sentiment).

**Data issues found and fixed.** (1) Customer keys use different formats (`151` vs `"A151"`) → normalised to integer `customer_id`. (2) The social file has 5 exact duplicate rows (dropped) and several rows per customer (up to 7, different platforms) → aggregated to one row per customer. (3) 10 missing `customer_rating` values (6.7 %) → median imputation plus a `rating_missing` flag. (4) Only 61 customers occur in both files.

**Join logic.** A naive row-level join produced 213 rows (96 phantom, 33 real transactions lost). We aggregated social data first and used a many-to-one **left join** from transactions (`validate='many_to_one'`), keeping all 150 transactions and flagging the 33 without social data (`has_social_profile`). The model-ready file contains the 117 transactions (61 customers) that have social context.

**Validation.** Row count preserved, `transaction_id` unique, social features present exactly for matched rows, ranges valid, hand-recomputed aggregate matches.

**Feature engineering (31 columns).** Aggregated social features (mean/max engagement and interest, platform count, dominant platform, sentiment shares/score, engagement×interest); temporal (month, weekday, weekend); amount tier; leakage-safe customer history (prior transaction count, prior average amount/rating, days since previous purchase, previous category).

**EDA findings.** Classes are balanced (25–35 per class); no IQR outliers in any numeric variable; engagement and interest are roughly uniform; sentiment skews positive (40/33/27 %); correlations between variables are weak (max |r| = 0.43, prior_txn_count vs month); product-level differences in social features are small with overlapping confidence intervals.

**Baseline signal check (grouped 5-fold CV by customer).** Majority baseline 0.205; Random Forest 0.20 (social only), 0.20 (social + history), 0.25 (incl. amount & rating) – close to the 0.20 chance level. We therefore evaluate with accuracy, macro-F1 and log-loss against the baseline and use customer-grouped splits to avoid leakage.

## Task 2 – Image processing
Each member submits `<member>_<neutral|smiling|surprised>.jpg`. Faces are detected (Haar cascade), cropped with a 25 % margin and resized to 224×224. Eight label-preserving augmentations are applied to every photo (±15° rotation, horizontal flip, grayscale, brighter, darker, Gaussian noise, blur). For each original and augmented image we extract 16-bin RGB histograms (48), HOG (324), uniform-LBP (10) and global statistics (≈12), optionally a 512-d ResNet-18 embedding, and save all rows to `image_features.csv` (9 rows per photo; `image_id` allows photo-level train/test splits to prevent leakage).

**Results on current photos (Nkuba, David, Rolande, Sheila).** 12 photos → 108 rows × 399 columns in `image_features.csv` (12 originals + 96 augmented); faces detected in 12/12 photos. PCA/silhouette analysis: expressions do not separate (silhouette ≈ 0); identity separates partly (silhouette 0.22 colour, 0.25 HOG) but Nkuba and Rolande separate easily while David and Sheila overlap – they were photographed in the most similar conditions – suggesting the hand-crafted features capture background and lighting as much as the face. We therefore recommend CNN embeddings, tight crops and consistent backgrounds for the Task 4 facial-recognition model, and image-level train/test splits to prevent augmentation leakage.
