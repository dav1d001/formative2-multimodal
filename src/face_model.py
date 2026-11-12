"""Train and run the Task 4 open-set facial recognition model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

try:
    from .model_utils import (
        FIG_DIR,
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
        FIG_DIR,
        MODELS_DIR,
        ROOT,
        evaluate,
        pick_unknown_threshold,
        save_confusion_matrix,
        save_json,
        unify_member,
    )

DATA_PATH = ROOT / "data" / "processed" / "image_features.csv"
META_COLUMNS = [
    "image_id",
    "member",
    "expression",
    "augmentation",
    "is_augmented",
    "face_detected",
    "source_file",
]
FINAL_MEMBERS = ["david", "junior", "rolande", "sheila"]


def _candidates():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
        ),
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


def _nearest_distances(train_x, test_x):
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    test_scaled = scaler.transform(test_x)
    return scaler, train_scaled, pairwise_distances(test_scaled, train_scaled).min(axis=1)


def train() -> dict:
    """Evaluate candidates by expression, calibrate unknown distance, and save the winner."""
    data = pd.read_csv(DATA_PATH)
    data["member"] = data["member"].map(unify_member)
    data["expression"] = data["expression"].astype(str).str.strip().str.lower()
    if sorted(data["member"].unique()) != sorted(FINAL_MEMBERS):
        raise ValueError(f"Expected members {FINAL_MEMBERS}; found {sorted(data['member'].unique())}.")
    feature_columns = [column for column in data.columns if column not in META_COLUMNS]
    x = data[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y = data["member"].to_numpy()
    expressions = data["expression"].to_numpy()
    members = data["member"].to_numpy()
    classes = np.array(FINAL_MEMBERS)

    comparison = {}
    pooled_predictions = {}
    for name, candidate in _candidates().items():
        pooled_proba = np.zeros((len(data), len(classes)), dtype=float)
        fold_results = []
        for expression in sorted(np.unique(expressions)):
            train_mask = expressions != expression
            test_mask = ~train_mask
            model = _candidates()[name]
            model.fit(x[train_mask], y[train_mask])
            probabilities = _aligned_proba(model, x[test_mask], classes)
            pooled_proba[test_mask] = probabilities
            metrics = evaluate(y[test_mask], probabilities, classes)
            metrics["held_out_expression"] = str(expression)
            fold_results.append(metrics)
        pooled_metrics = evaluate(y, pooled_proba, classes)
        comparison[name] = {"folds": fold_results, "pooled": pooled_metrics}
        pooled_predictions[name] = (pooled_proba, classes[np.argmax(pooled_proba, axis=1)])

    best_name = min(
        comparison,
        key=lambda name: (-comparison[name]["pooled"]["macro_f1"], comparison[name]["pooled"]["log_loss"]),
    )
    known_distances = []
    unknown_distances = []
    for expression in sorted(np.unique(expressions)):
        train_mask = expressions != expression
        test_mask = ~train_mask
        _, _, distances = _nearest_distances(x[train_mask], x[test_mask])
        known_distances.extend(distances.tolist())
    for member in sorted(np.unique(members)):
        train_mask = members != member
        test_mask = ~train_mask
        _, _, distances = _nearest_distances(x[train_mask], x[test_mask])
        unknown_distances.extend(distances.tolist())
    threshold = pick_unknown_threshold(known_distances, unknown_distances, higher_is_known=False)
    distance_predictions = np.array(known_distances + unknown_distances)
    distance_truth = np.array([1] * len(known_distances) + [0] * len(unknown_distances))
    distance_authorized = distance_predictions <= threshold
    unknown_metrics = {
        "threshold": threshold,
        "known_distance_count": len(known_distances),
        "unknown_distance_count": len(unknown_distances),
        "balanced_accuracy": float(balanced_accuracy_score(distance_truth, distance_authorized)),
        "known_mean_distance": float(np.mean(known_distances)),
        "unknown_mean_distance": float(np.mean(unknown_distances)),
    }

    best_model = _candidates()[best_name]
    best_model.fit(x, y)
    ref_scaler = StandardScaler().fit(x)
    reference_matrix = ref_scaler.transform(x)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "classes": classes.tolist(),
            "ref_scaler": ref_scaler,
            "reference_matrix": reference_matrix,
            "unknown_threshold": threshold,
            "unknown_rule": "accept iff nearest standardized training-row distance <= threshold and a face was detected",
        },
        MODELS_DIR / "face_model.joblib",
    )
    best_proba, best_pred = pooled_predictions[best_name]
    save_confusion_matrix(y, best_pred, classes, "Face recognition: leave-one-expression-out", "face_confusion_matrix.png")
    metrics_report = {
        "members": classes.tolist(),
        "features": feature_columns,
        "selection": "highest pooled macro-F1, then lowest pooled log-loss",
        "candidates": comparison,
        "selected_model": best_name,
        "unknown_rule": unknown_metrics,
    }
    save_json(metrics_report, "face_metrics.json")
    print("Face model comparison (pooled expression-held-out metrics):")
    print(pd.DataFrame({name: entry["pooled"] for name, entry in comparison.items()}).T.to_string())
    print(f"Selected: {best_name}; unknown threshold={threshold:.4f}; distance balanced accuracy={unknown_metrics['balanced_accuracy']:.3f}")
    return metrics_report


def predict_image(image_path: str | Path) -> dict:
    """Classify an image and reject it unless its nearest-row distance passes the open-set rule."""
    try:
        from . import image_pipeline
    except ImportError:
        import image_pipeline
    artifact = joblib.load(MODELS_DIR / "face_model.joblib")
    image, face_detected = image_pipeline.load_face(str(image_path))
    feature_row = pd.DataFrame([image_pipeline.extract_features(image)]).reindex(
        columns=artifact["feature_columns"], fill_value=0
    )
    features = feature_row.to_numpy(dtype=float)
    probabilities = _aligned_proba(artifact["model"], features, artifact["classes"])[0]
    guess_index = int(np.argmax(probabilities))
    standardized = artifact["ref_scaler"].transform(features)
    distance = float(pairwise_distances(standardized, artifact["reference_matrix"]).min())
    authorized = bool(face_detected and distance <= artifact["unknown_threshold"])
    return {
        "user": artifact["classes"][guess_index] if authorized else None,
        "best_guess": artifact["classes"][guess_index],
        "confidence": float(probabilities[guess_index]),
        "distance": distance,
        "authorized": authorized,
        "face_detected": bool(face_detected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", help="Predict and open-set-check one image")
    args = parser.parse_args()
    if args.predict:
        print(predict_image(args.predict))
    else:
        train()


if __name__ == "__main__":
    main()