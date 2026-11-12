"""Train a leakage-aware product model and demonstrate simulated recommendations."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from .model_utils import MODELS_DIR, ROOT, evaluate, save_confusion_matrix, save_json, unify_member
except ImportError:
    from model_utils import MODELS_DIR, ROOT, evaluate, save_confusion_matrix, save_json, unify_member

DATA_PATH = ROOT / "data" / "processed" / "merged_dataset_model_ready.csv"
TARGET = "product_category"
GROUP = "customer_id"
ID_COLUMNS = ["customer_id", "transaction_id", "purchase_date"]
POST_PURCHASE_COLUMNS = ["purchase_amount", "customer_rating", "rating_missing", "amount_tier"]
FINAL_MEMBERS = ["david", "junior", "rolande", "sheila"]


def _preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="none")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        remainder="drop",
    )


def _estimators(frame: pd.DataFrame, features: list[str], class_count: int):
    def pipeline(classifier):
        return Pipeline([("preprocess", _preprocessor(frame, features)), ("classifier", classifier)])

    return {
        "baseline_prior": pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)),
        "logistic_regression_c0.05": pipeline(
            LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", random_state=0)
        ),
        "random_forest": pipeline(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
        "xgboost": pipeline(XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            n_jobs=1,
            random_state=0,
            verbosity=0,
        )),
    }


def _aligned_proba(model, values, class_count: int):
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), class_count), dtype=float)
    for column, label in enumerate(model.named_steps["classifier"].classes_):
        aligned[:, int(label)] = raw[:, column]
    return aligned


def _run_repeated_cv(frame, features, y, y_codes, groups, class_names, seeds=range(5)):
    fold_metrics = {name: [] for name in _estimators(frame, features, len(class_names))}
    probability_sums = {name: np.zeros((len(frame), len(class_names)), dtype=float) for name in fold_metrics}
    prediction_counts = np.zeros(len(frame), dtype=int)
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_indices, test_indices) in enumerate(splitter.split(frame[features], y_codes, groups)):
            prediction_counts[test_indices] += 1
            for name, estimator in _estimators(frame, features, len(class_names)).items():
                estimator.fit(frame.iloc[train_indices][features], y_codes[train_indices])
                probabilities = _aligned_proba(estimator, frame.iloc[test_indices][features], len(class_names))
                probability_sums[name][test_indices] += probabilities
                metrics = evaluate(y[test_indices], probabilities, class_names)
                metrics.update({"seed": int(seed), "fold": fold + 1})
                fold_metrics[name].append(metrics)
    if not np.all(prediction_counts == len(list(seeds))):
        raise RuntimeError("Repeated grouped CV did not produce one out-of-fold prediction per seed and row.")
    mean_probabilities = {
        name: values / prediction_counts[:, None] for name, values in probability_sums.items()
    }
    summary = {}
    for name, rows in fold_metrics.items():
        table = pd.DataFrame(rows)
        summary[name] = {
            "mean": {metric: float(table[metric].mean()) for metric in ("accuracy", "macro_f1", "log_loss")},
            "std": {metric: float(table[metric].std(ddof=1)) for metric in ("accuracy", "macro_f1", "log_loss")},
            "folds": rows,
        }
    return summary, mean_probabilities


def train() -> dict:
    """Evaluate product categories using repeated customer-grouped cross-validation."""
    data = pd.read_csv(DATA_PATH)
    feature_columns = [
        column for column in data.columns
        if column not in [TARGET] + ID_COLUMNS + POST_PURCHASE_COLUMNS
    ]
    required = {TARGET, GROUP, *feature_columns}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing required product columns: {sorted(required - set(data.columns))}")
    y = data[TARGET].astype(str).to_numpy()
    label_encoder = LabelEncoder().fit(y)
    y_codes = label_encoder.transform(y)
    class_names = label_encoder.classes_.tolist()
    groups = data[GROUP].to_numpy()
    comparison, probabilities = _run_repeated_cv(
        data, feature_columns, y, y_codes, groups, class_names, seeds=range(5)
    )
    baseline_log_loss = comparison["baseline_prior"]["mean"]["log_loss"]
    log_loss_ceiling = baseline_log_loss + 0.05
    standard_candidates = ["logistic_regression", "random_forest", "xgboost"]
    eligible_models = [
        name for name in standard_candidates
        if comparison[name]["mean"]["log_loss"] <= log_loss_ceiling
    ]
    if eligible_models:
        best_name = max(eligible_models, key=lambda name: comparison[name]["mean"]["macro_f1"])
        selection_reason = "highest mean macro-F1 among models within 0.05 of baseline mean log-loss"
    else:
        best_name = "logistic_regression_c0.05"
        selection_reason = (
            "the best macro-F1 standard candidate exceeded the baseline log-loss limit; "
            "selected LogisticRegression(C=0.05), whose mean log-loss is within the limit"
        )

    best_model = _estimators(data, feature_columns, len(class_names))[best_name]
    best_model.fit(data[feature_columns], y_codes)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "classes": class_names,
            "label_encoder": label_encoder,
            "best_model": best_name,
        },
        MODELS_DIR / "product_model.joblib",
    )

    mapping_customers = (
        data.groupby(GROUP).size().rename("transactions").reset_index()
        .sort_values(["transactions", GROUP], ascending=[False, True])
        .head(len(FINAL_MEMBERS))[GROUP]
        .tolist()
    )
    member_mapping = {
        member: int(customer)
        for member, customer in zip(FINAL_MEMBERS, mapping_customers)
    }
    save_json(member_mapping, "member_to_customer.json")

    post_columns = [column for column in POST_PURCHASE_COLUMNS if column in data.columns]
    ablation_features = feature_columns + post_columns
    ablation_summary, _ = _run_repeated_cv(
        data, ablation_features, y, y_codes, groups, class_names, seeds=range(5)
    )
    ablation_model = ablation_summary[best_name]
    save_confusion_matrix(
        y,
        np.asarray(class_names)[np.argmax(probabilities[best_name], axis=1)],
        class_names,
        "Product category: repeated customer-grouped out-of-fold predictions",
        "product_confusion_matrix.png",
    )
    metrics_report = {
        "classes": class_names,
        "features": feature_columns,
        "excluded_id_columns": ID_COLUMNS,
        "excluded_post_purchase_columns": POST_PURCHASE_COLUMNS,
        "split": "StratifiedGroupKFold(n_splits=5, shuffle=True), seeds 0-4, grouped by customer_id",
        "comparison": comparison,
        "selected_model": best_name,
        "selection": {
            "baseline_mean_log_loss": baseline_log_loss,
            "maximum_eligible_mean_log_loss": log_loss_ceiling,
            "eligible_models": eligible_models,
            "criterion": "highest mean macro-F1 among standard models with mean log-loss <= baseline + 0.05",
            "fallback_model": "LogisticRegression(C=0.05)",
            "reason": selection_reason,
        },
        "selected_out_of_fold_metrics": evaluate(y, probabilities[best_name], class_names),
        "ablation_not_allowed_for_real_prediction": {
            "features_added": post_columns,
            "model": best_name,
            "mean": ablation_model["mean"],
            "std": ablation_model["std"],
        },
        "member_to_customer_simulation": member_mapping,
    }
    save_json(metrics_report, "product_metrics.json")

    summary_rows = []
    for name, entry in comparison.items():
        summary_rows.append({
            "model": name,
            **{f"{metric}_mean": entry["mean"][metric] for metric in ("accuracy", "macro_f1", "log_loss")},
            **{f"{metric}_std": entry["std"][metric] for metric in ("accuracy", "macro_f1", "log_loss")},
        })
    summary_table = pd.DataFrame(summary_rows).set_index("model")
    print("Product grouped-CV results (mean +/- std across 25 folds):")
    print(summary_table.to_string(float_format=lambda value: f"{value:.3f}"))
    print("Baseline prior next to selected model:")
    print(summary_table.loc[["baseline_prior", best_name]].to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"Selection rule: {selection_reason}; log-loss ceiling={log_loss_ceiling:.3f}.")
    baseline_f1 = comparison["baseline_prior"]["mean"]["macro_f1"]
    best_f1 = comparison[best_name]["mean"]["macro_f1"]
    accuracy_gain = (
        comparison[best_name]["mean"]["accuracy"]
        - comparison["baseline_prior"]["mean"]["accuracy"]
    )
    print(
        f"Selected non-baseline model: {best_name}; macro-F1 gain={best_f1 - baseline_f1:.3f}, "
        f"accuracy gain={accuracy_gain:.3f}."
    )
    if accuracy_gain < 0.05 or best_f1 - baseline_f1 < 0.05:
        print("Honest note: the gain over baseline is small; social and transaction-time signals are weak for this target.")
    print("Ablation with post-purchase columns (NOT allowed for a real prediction):")
    print(pd.DataFrame([{
        "model": best_name,
        "accuracy_mean": ablation_model["mean"]["accuracy"],
        "macro_f1_mean": ablation_model["mean"]["macro_f1"],
        "log_loss_mean": ablation_model["mean"]["log_loss"],
        "added_columns": ", ".join(post_columns),
    }]).to_string(index=False))
    print("Member-to-customer assignment is a simulation, not a claim that team members are customers.")
    return metrics_report


def recommend(member: str, top_k: int = 3) -> dict:
    """Recommend using a simulated member/customer pairing; team members are not customers."""
    artifact = joblib.load(MODELS_DIR / "product_model.joblib")
    mapping = pd.read_json(MODELS_DIR / "member_to_customer.json", typ="series").to_dict()
    normalized = unify_member(member)
    if normalized not in mapping:
        raise ValueError(f"Unknown member {normalized!r}; expected one of {FINAL_MEMBERS}.")
    data = pd.read_csv(DATA_PATH)
    customer_id = mapping[normalized]
    customer_rows = data.loc[data[GROUP] == customer_id].copy()
    if customer_rows.empty:
        raise ValueError(f"Simulated customer {customer_id} was not found in the product data.")
    if "transaction_id" in customer_rows:
        customer_rows = customer_rows.sort_values("transaction_id")
    feature_row = customer_rows.iloc[[-1]][artifact["feature_columns"]]
    probabilities = artifact["model"].predict_proba(feature_row)[0]
    order = np.argsort(probabilities)[::-1][:max(1, int(top_k))]
    recommendations = [
        {"product_category": artifact["classes"][index], "probability": float(probabilities[index])}
        for index in order
    ]
    return {
        "member": normalized,
        "simulated_customer_id": int(customer_id),
        "prediction": recommendations[0]["product_category"],
        "top_k": recommendations,
        "simulation": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommend", help="Recommend products for a simulated team-member profile")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.recommend:
        print(recommend(args.recommend, args.top_k))
    else:
        train()


if __name__ == "__main__":
    main()