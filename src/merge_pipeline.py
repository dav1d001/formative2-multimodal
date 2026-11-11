"""
Task 1 - Data merge + feature engineering
=========================================
Inputs  : data/raw/customer_transactions.xlsx, data/raw/customer_social_profiles.xlsx
Outputs : data/processed/merged_dataset.csv            (every transaction, 1 row each)
          data/processed/merged_dataset_model_ready.csv (only transactions with a social profile)

Design decisions (justified in the notebook / report):
  1. IDs differ in format ('A178' vs 178) -> normalise both to an integer `customer_id`.
  2. Social data is NOT one-row-per-customer (155 rows, 84 customers, 5 exact duplicates)
     -> drop exact duplicates, then AGGREGATE to one row per customer *before* joining,
        otherwise the join fans out and silently inflates / duplicates transactions.
  3. LEFT join from transactions (the prediction target `product_category` lives there),
     with an explicit `has_social_profile` flag instead of silently losing transactions.
  4. History features use only *earlier* transactions (shifted expanding stats) to avoid leakage.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SENT_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}


def load_and_clean_transactions(path=RAW / "customer_transactions.xlsx"):
    t = pd.read_excel(path)
    t = t.rename(columns={"customer_id_legacy": "customer_id"})
    t["customer_id"] = t["customer_id"].astype(int)
    t["purchase_date"] = pd.to_datetime(t["purchase_date"])
    t["product_category"] = t["product_category"].str.strip().str.title()
    assert t["transaction_id"].is_unique, "transaction_id must be unique"
    # missing rating: keep an indicator, then impute with the global median (robust to skew)
    t["rating_missing"] = t["customer_rating"].isna().astype(int)
    t["customer_rating"] = t["customer_rating"].fillna(t["customer_rating"].median())
    return t.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)


def load_and_clean_social(path=RAW / "customer_social_profiles.xlsx"):
    s = pd.read_excel(path)
    n_raw = len(s)
    s["customer_id"] = s["customer_id_new"].str.extract(r"^A(\d+)$", expand=False).astype(int)
    s["social_media_platform"] = s["social_media_platform"].str.strip()   # keep brand casing (LinkedIn, TikTok)
    s["review_sentiment"] = s["review_sentiment"].str.strip().str.title()
    s = s.drop(columns="customer_id_new").drop_duplicates().reset_index(drop=True)
    s.attrs["n_exact_duplicates_dropped"] = n_raw - len(s)
    return s


def aggregate_social(s):
    """Collapse to exactly one row per customer."""
    s = s.assign(sentiment_score=s["review_sentiment"].map(SENT_MAP))
    g = s.groupby("customer_id")
    agg = g.agg(
        n_social_records=("social_media_platform", "size"),
        n_platforms=("social_media_platform", "nunique"),
        avg_engagement=("engagement_score", "mean"),
        max_engagement=("engagement_score", "max"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        max_purchase_interest=("purchase_interest_score", "max"),
        avg_sentiment_score=("sentiment_score", "mean"),
    )
    # dominant platform = the one with highest total engagement (deterministic tie-break)
    plat = (s.groupby(["customer_id", "social_media_platform"])["engagement_score"].sum()
              .reset_index().sort_values(["customer_id", "engagement_score", "social_media_platform"],
                                         ascending=[True, False, True])
              .drop_duplicates("customer_id").set_index("customer_id")["social_media_platform"])
    agg["top_platform"] = plat
    for lab in ["Positive", "Neutral", "Negative"]:
        agg[f"share_{lab.lower()}"] = g["review_sentiment"].apply(lambda x, l=lab: (x == l).mean())
    agg["dominant_sentiment"] = agg["avg_sentiment_score"].apply(
        lambda v: "Positive" if v > 0.2 else ("Negative" if v < -0.2 else "Neutral"))
    agg["engagement_x_interest"] = agg["avg_engagement"] * agg["avg_purchase_interest"] / 100
    return agg.round(4).reset_index()


def add_transaction_features(df):
    df = df.sort_values(["purchase_date", "transaction_id"]).copy()
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    df["amount_tier"] = pd.qcut(df["purchase_amount"], 3, labels=["Low", "Mid", "High"]).astype(str)
    g = df.groupby("customer_id")
    # --- leakage-safe customer history (strictly earlier transactions only) ---
    df["prior_txn_count"] = g.cumcount()
    df["prior_avg_amount"] = g["purchase_amount"].transform(lambda x: x.shift().expanding().mean())
    df["prior_avg_rating"] = g["customer_rating"].transform(lambda x: x.shift().expanding().mean())
    df["days_since_prev_txn"] = g["purchase_date"].diff().dt.days
    df["prev_category"] = g["product_category"].shift()
    df["is_repeat_customer"] = (df["prior_txn_count"] > 0).astype(int)
    for c in ["prior_avg_amount", "prior_avg_rating", "days_since_prev_txn"]:
        df[c] = df[c].round(3)
    df["prev_category"] = df["prev_category"].fillna("No_Prior")
    return df


def build():
    t = load_and_clean_transactions()
    s_clean = load_and_clean_social()
    s_agg = aggregate_social(s_clean)
    assert s_agg["customer_id"].is_unique
    merged = t.merge(s_agg, on="customer_id", how="left", validate="many_to_one", indicator=True)
    merged["has_social_profile"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns="_merge")
    merged = add_transaction_features(merged)
    # post-merge integrity checks
    assert len(merged) == len(t), "left join must preserve transaction count"
    assert merged["transaction_id"].is_unique
    return t, s_clean, s_agg, merged.sort_values("transaction_id").reset_index(drop=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t, s_clean, s_agg, merged = build()
    merged.to_csv(OUT / "merged_dataset.csv", index=False)
    ready = merged[merged["has_social_profile"] == 1].reset_index(drop=True)
    # first-ever purchases have no history: 0 + `is_repeat_customer` flag keeps them model-friendly
    ready[["prior_avg_amount", "prior_avg_rating", "days_since_prev_txn"]] = \
        ready[["prior_avg_amount", "prior_avg_rating", "days_since_prev_txn"]].fillna(0)
    ready.to_csv(OUT / "merged_dataset_model_ready.csv", index=False)
    print(f"transactions={len(t)} | social(clean)={len(s_clean)} | social customers={len(s_agg)}")
    print(f"merged={merged.shape} | with social={len(ready)} | customers in model-ready={ready.customer_id.nunique()}")
