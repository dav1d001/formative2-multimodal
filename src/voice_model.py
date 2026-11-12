"""Train Task 4 voice identification and claimed-speaker verification models."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

try:
    from .model_utils import (
        MODELS_DIR,
        ROOT,
        evaluate,
        pick_unknown_threshold,
        save_confusion_matrix,
        save_json,
        unify_member,
    )
except ImportError:
    from model_utils import (
        MODELS_DIR,
        ROOT,
        evaluate,
        pick_unknown_threshold,
        save_confusion_matrix,
        save_json,
        unify_member,
    )

AUDIO_PATH = ROOT / "task3_audio" / "audio_features.csv"
OUTSIDER_PATH = ROOT / "task3_audio" / "unauthorized_audio_features.csv"
NON_FEATURE_COLUMNS = ["file", "member", "phrase", "take", "augmentation", "duration_s"]
FINAL_MEMBERS = ["david", "junior", "rolande", "sheila"]


def _candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
        "svc_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        ),
    }


def _aligned_proba(model, values, classes):
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), len(classes)), dtype=float)
    class_positions = {label: index for index, label in enumerate(classes)}
    for column, label in enumerate(model.classes_):
        aligned[:, class_positions[label]] = raw[:, column]
    return aligned


def _split_groups(data: pd.DataFrame) -> tuple[str, list[str]]:
    take_counts = data.groupby(["member", "phrase"])["take"].nunique()
    if len(take_counts) and (take_counts >= 2).all():
        return "take", sorted(data["take"].astype(str).unique())
    return "phrase", sorted(data["phrase"].astype(str).unique())


def _feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Build the requested feature groups using lowercase column-name rules."""
    excluded = {column.lower() for column in NON_FEATURE_COLUMNS}
    numeric = [
        column for column in data.columns
        if column.lower() not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]
    lower_names = {column: column.lower() for column in numeric}
    mfcc_means = [
        column for column in numeric
        if lower_names[column].startswith("mfcc") and lower_names[column].endswith("_mean")
    ]
    spectral = [
        column for column in numeric
        if lower_names[column].startswith(("centroid", "bandwidth", "rolloff"))
        and lower_names[column].endswith(("_mean", "_std"))
    ]
    return {
        "all": numeric,
        "mfcc_means": mfcc_means,
        "mfcc_means+spectral_means": list(dict.fromkeys(mfcc_means + spectral)),
    }


def _verification_metrics(y, probabilities, classes) -> dict:
    """Calibrate the claimed-speaker threshold from out-of-fold known data only."""
    class_indices = {label: index for index, label in enumerate(classes)}
    true_indices = np.array([class_indices[label] for label in y])
    genuine_scores = probabilities[np.arange(len(y)), true_indices]
    impostor_scores = np.concatenate(
        [np.delete(row, true_index) for row, true_index in zip(probabilities, true_indices)]
    )
    threshold = pick_unknown_threshold(genuine_scores, impostor_scores, higher_is_known=True)
    verification_truth = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    accepted = np.concatenate([genuine_scores, impostor_scores]) >= threshold
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(verification_truth, accepted)),
        "genuine_score_count": len(genuine_scores),
        "impostor_score_count": len(impostor_scores),
    }


def _cross_validate_pair(data, feature_columns, y, classes, fold_groups, groups, model_name):
    """Run phrase/take-held-out identification and verification for one pair."""
    values = data[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    pooled_proba = np.zeros((len(data), len(classes)), dtype=float)
    fold_results = []
    for group in groups:
        train_mask = fold_groups != group
        test_mask = ~train_mask
        model = _candidates()[model_name]
        model.fit(values[train_mask], y[train_mask])
        probabilities = _aligned_proba(model, values[test_mask], classes)
        pooled_proba[test_mask] = probabilities
        metrics = evaluate(y[test_mask], probabilities, classes)
        metrics["held_out_group"] = str(group)
        fold_results.append(metrics)
    return {
        "folds": fold_results,
        "pooled": evaluate(y, pooled_proba, classes),
        "verification": _verification_metrics(y, pooled_proba, classes),
    }, pooled_proba


def train() -> dict:
    """Compare voice feature groups and models using recording-grouped cross-validation."""
    data = pd.read_csv(AUDIO_PATH)
    data["member"] = data["member"].map(unify_member)
    data["phrase"] = data["phrase"].astype(str).str.strip().str.lower()
    data["take"] = data["take"].astype(str)
    classes = np.array(FINAL_MEMBERS)
    if sorted(data["member"].unique()) != sorted(FINAL_MEMBERS):
        raise ValueError(f"Expected members {FINAL_MEMBERS}; found {sorted(data['member'].unique())}.")
    group_column, groups = _split_groups(data)
    if group_column == "take":
        data["fold_group"] = data["phrase"].astype(str) + "_take_" + data["take"].astype(str)
        groups = sorted(data["fold_group"].unique())
    else:
        data["fold_group"] = data["phrase"]
    feature_sets = _feature_sets(data)
    y = data["member"].to_numpy()
    fold_groups = data["fold_group"].to_numpy()

    ablation = []
    pooled_probabilities = {}
    for feature_set, feature_columns in feature_sets.items():
        for model_name in _candidates():
            result, probabilities = _cross_validate_pair(
                data, feature_columns, y, classes, fold_groups, groups, model_name
            )
            ablation.append({
                "feature_set": feature_set,
                "model": model_name,
                "feature_count": len(feature_columns),
                **result,
            })
            pooled_probabilities[(feature_set, model_name)] = probabilities

    selected = min(
        ablation,
        key=lambda result: (-result["pooled"]["macro_f1"], result["pooled"]["log_loss"]),
    )
    selected_feature_set = selected["feature_set"]
    selected_model_name = selected["model"]
    feature_columns = feature_sets[selected_feature_set]
    best_proba = pooled_probabilities[(selected_feature_set, selected_model_name)]

    model = _candidates()[selected_model_name]
    model.fit(
        data[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float),
        y,
    )
    threshold = selected["verification"]["threshold"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "classes": classes.tolist(),
            "threshold": threshold,
            "model_name": selected_model_name,
            "feature_set": selected_feature_set,
            "split": group_column,
            "chance_accuracy": 1 / len(classes),
        },
        MODELS_DIR / "voice_model.joblib",
    )

    outsider_report = []
    outsider = pd.read_csv(OUTSIDER_PATH)
    outsider["member"] = outsider["member"].map(unify_member)
    outsider_features = outsider.reindex(columns=feature_columns, fill_value=0).apply(pd.to_numeric, errors="raise")
    outsider_probabilities = _aligned_proba(model, outsider_features.to_numpy(dtype=float), classes)[0]
    for index, claimed in enumerate(classes):
        score = float(outsider_probabilities[index])
        outsider_report.append({
            "claimed_identity": claimed,
            "probability": score,
            "rejected": bool(score < threshold),
        })

    save_confusion_matrix(
        y,
        classes[np.argmax(best_proba, axis=1)],
        classes,
        f"Voice identification: leave-one-{group_column}-out",
        "voice_confusion_matrix.png",
    )
    metrics_report = {
        "members": classes.tolist(),
        "features": feature_columns,
        "feature_sets": feature_sets,
        "split": f"leave-one-{group_column}-out",
        "chance_accuracy": 1 / len(classes),
        "feature_ablation": ablation,
        "candidates": {
            result["model"]: {"folds": result["folds"], "pooled": result["pooled"]}
            for result in ablation
            if result["feature_set"] == selected_feature_set
        },
        "selected_model": selected_model_name,
        "selected_feature_set": selected_feature_set,
        "verification": selected["verification"],
        "outsider_demo_not_used_for_tuning": outsider_report,
    }
    save_json(metrics_report, "voice_metrics.json")
    table = pd.DataFrame([
        {
            "feature_set": result["feature_set"],
            "model": result["model"],
            "feature_count": result["feature_count"],
            **result["pooled"],
            "verification_balanced_accuracy": result["verification"]["balanced_accuracy"],
        }
        for result in ablation
    ]).drop(columns=["held_out_group"], errors="ignore")
    print(f"Voice feature/model ablation (pooled leave-one-{group_column}-out; chance accuracy={1 / len(classes):.3f}):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"Selected: {selected_feature_set} + {selected_model_name}; "
        f"verification threshold={threshold:.4f}; balanced accuracy={selected['verification']['balanced_accuracy']:.3f}"
    )
    if selected["verification"]["balanced_accuracy"] < 0.8:
        print("WARNING: voice verification balanced accuracy is below 0.8; record more takes per member and phrase.")
    print("Untuned outsider demonstration (one row; descriptive only):")
    print(pd.DataFrame(outsider_report).to_string(index=False))
    return metrics_report


def predict_features(feature_row_df: pd.DataFrame, claimed_user: str | None = None) -> dict:
    """Verify a voice feature row against the claimed identity, or its best-guess identity."""
    artifact = joblib.load(MODELS_DIR / "voice_model.joblib")
    row = feature_row_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    row = row.reindex(columns=artifact["feature_columns"], fill_value=0).apply(pd.to_numeric, errors="raise")
    probabilities = _aligned_proba(artifact["model"], row.to_numpy(dtype=float), artifact["classes"])[0]
    guess_index = int(np.argmax(probabilities))
    best_guess = artifact["classes"][guess_index]
    claimed = unify_member(claimed_user) if claimed_user is not None else best_guess
    if claimed not in artifact["classes"]:
        score = 0.0
        authorized = False
    else:
        score = float(probabilities[artifact["classes"].index(claimed)])
        authorized = score >= artifact["threshold"]
    return {
        "claimed_user": claimed,
        "best_guess": best_guess,
        "confidence": score,
        "authorized": bool(authorized),
        "probabilities": dict(zip(artifact["classes"], probabilities.astype(float).tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", help="CSV containing one voice feature row")
    parser.add_argument("--claimed", help="Identity claimed by the face step")
    args = parser.parse_args()
    if args.predict:
        print(predict_features(pd.read_csv(args.predict), args.claimed))
    else:
        train()


if __name__ == "__main__":
    main()