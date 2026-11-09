"""
Task 2 - Facial image collection, augmentation and feature extraction
=====================================================================
Expected input  : data/images/<member>_<expression>.jpg      e.g.  amara_neutral.jpg
                  expression in {neutral, smiling, surprised}  (aliases smile/surprise are normalised)
Outputs         : data/images_augmented/*.jpg                (every augmented image, for inspection)
                  data/processed/image_features.csv          (1 row per original + per augmented image)

Pipeline per image:  load -> detect/crop face (Haar cascade, fallback = centre crop) -> resize 224x224
                     -> 8 augmentations -> features (RGB histogram, HOG, LBP, stats [, CNN embedding])
"""
from __future__ import annotations
import re, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
from skimage.feature import hog, local_binary_pattern

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 224
EXPRESSIONS = ["neutral", "smiling", "surprised"]
ALIASES = {"smile": "smiling", "smiling": "smiling", "happy": "smiling",
           "surprise": "surprised", "surprised": "surprised", "neutral": "neutral"}
NAME_RE = re.compile(r"^(?P<member>.+)_(?P<expr>[A-Za-z]+)\.(jpg|jpeg|png)$", re.I)   # expression = text after LAST underscore

def _load_detector():
    """Haar cascade ships with opencv-python 4.x; OpenCV 5 removed it -> fall back to centre-crop."""
    if hasattr(cv2, "CascadeClassifier"):
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    warnings.warn("cv2.CascadeClassifier unavailable (OpenCV 5?). Face cropping disabled -> centre crop. "
                  "Use `pip install 'opencv-python>=4.8,<5'` for automatic face detection.")
    return None

_FACE = _load_detector()


# ----------------------------------------------------------------------------- discovery & QA
def discover_images(images_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(images_dir).glob("*")):
        m = NAME_RE.match(p.name)
        if not m:
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                warnings.warn(f"Skipping '{p.name}': expected <member>_<expression>.jpg")
            continue
        expr = ALIASES.get(m["expr"].lower())
        if expr is None:
            warnings.warn(f"Skipping '{p.name}': unknown expression '{m['expr']}'")
            continue
        rows.append(dict(member=m["member"].lower(), expression=expr, path=str(p), file=p.name))
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Member x expression completeness table (rubric: each member = neutral + smile + surprised)."""
    if df.empty:
        raise FileNotFoundError("No valid images found. Name files like amara_neutral.jpg")
    tab = df.pivot_table(index="member", columns="expression", values="file", aggfunc="count", fill_value=0)
    tab = tab.reindex(columns=EXPRESSIONS, fill_value=0)
    tab["complete"] = (tab[EXPRESSIONS] >= 1).all(axis=1)
    return tab


# ----------------------------------------------------------------------------- loading
def load_face(path: str, size: int = IMG_SIZE, margin: float = 0.25):
    """Return (RGB uint8 size x size, face_detected: bool)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(60, w // 8),) * 2) \
        if _FACE is not None else []
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        side = int(max(fw, fh) * (1 + 2 * margin)); cx, cy = x + fw // 2, y + fh // 2
        detected = True
    else:                                                  # fallback: centre square crop
        side, cx, cy, detected = min(h, w), w // 2, h // 2, False
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = bgr[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detected


# ----------------------------------------------------------------------------- augmentations
def _rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def _gray(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

def _noise(img, sigma=12, seed=42):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

AUGMENTATIONS = {
    "rot_plus15":   lambda im: _rotate(im, 15),
    "rot_minus15":  lambda im: _rotate(im, -15),
    "flip_h":       lambda im: cv2.flip(im, 1),
    "grayscale":    _gray,
    "brighter":     lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=40),
    "darker":       lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=-40),
    "gauss_noise":  _noise,
    "blur":         lambda im: cv2.GaussianBlur(im, (7, 7), 0),
}


# ----------------------------------------------------------------------------- features
def extract_features(img_rgb: np.ndarray) -> dict:
    f = {}
    # 1) colour histograms: 16 bins x R,G,B, L1-normalised
    for i, ch in enumerate("rgb"):
        h = cv2.calcHist([img_rgb], [i], None, [16], [0, 256]).ravel()
        h /= h.sum() + 1e-9
        f.update({f"hist_{ch}_{b:02d}": v for b, v in enumerate(h)})
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # 2) HOG on 128x128 grayscale (captures face shape / expression edges)
    g128 = cv2.resize(gray, (128, 128))
    hv = hog(g128, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    f.update({f"hog_{i:03d}": v for i, v in enumerate(hv)})
    # 3) LBP texture histogram (uniform, P=8, R=1 -> 10 bins)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lh, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    f.update({f"lbp_{i:02d}": v for i, v in enumerate(lh)})
    # 4) global statistics
    for i, ch in enumerate("rgb"):
        f[f"mean_{ch}"] = float(img_rgb[..., i].mean()); f[f"std_{ch}"] = float(img_rgb[..., i].std())
    f["brightness"] = float(gray.mean()); f["contrast"] = float(gray.std())
    f["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    f["edge_density"] = float((cv2.Canny(gray, 100, 200) > 0).mean())
    return f


def cnn_embedder():
    """Optional 512-d ResNet-18 embedding (needs `pip install torch torchvision`). Returns fn or None."""
    try:
        import torch, torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity(); m.eval()
        mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
        def embed(img_rgb):
            x = ((img_rgb.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
            with torch.no_grad():
                return {f"emb_{i:03d}": float(v) for i, v in enumerate(m(torch.from_numpy(x))[0].numpy())}
        return embed
    except Exception as e:                                     # torch missing / weights unavailable
        warnings.warn(f"CNN embeddings disabled ({type(e).__name__}); using histogram/HOG/LBP features only.")
        return None


# ----------------------------------------------------------------------------- orchestration
def run_pipeline(images_dir=ROOT / "data" / "images", out_csv=ROOT / "data" / "processed" / "image_features.csv",
                 aug_dir=ROOT / "data" / "images_augmented", use_embeddings=False, augmentations=AUGMENTATIONS):
    inv = discover_images(images_dir); report = validate_dataset(inv)
    embed = cnn_embedder() if use_embeddings else None
    Path(aug_dir).mkdir(parents=True, exist_ok=True); Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in inv.itertuples():
        img, det = load_face(r.path)
        stem = Path(r.file).stem
        variants = {"original": img} | {k: fn(img) for k, fn in augmentations.items()}
        for aug, im in variants.items():
            if aug != "original":
                cv2.imwrite(str(Path(aug_dir) / f"{stem}__{aug}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            row = dict(image_id=stem, member=r.member, expression=r.expression, augmentation=aug,
                       is_augmented=int(aug != "original"), face_detected=int(det), source_file=r.file)
            row.update(extract_features(im))
            if embed: row.update(embed(im))
            rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    return df, report, inv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=str(ROOT / "data" / "images"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "image_features.csv"))
    ap.add_argument("--aug-dir", default=str(ROOT / "data" / "images_augmented"))
    ap.add_argument("--embeddings", action="store_true", help="add 512-d ResNet-18 embeddings (needs torch)")
    a = ap.parse_args()
    df, report, _ = run_pipeline(a.images_dir, a.out, a.aug_dir, a.embeddings)
    print(report.to_string()); print(f"\nimage_features.csv -> {df.shape[0]} rows x {df.shape[1]} cols "
          f"({df.is_augmented.eq(0).sum()} originals + {df.is_augmented.sum()} augmented)")
