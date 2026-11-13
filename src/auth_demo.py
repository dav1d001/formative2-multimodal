import argparse
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import cv2
import joblib
import numpy as np

import audio_pipeline
import face_model
import product_model
import voice_model
from model_utils import MODELS_DIR, unify_member

IMAGES = ROOT / "data" / "images"
OUTSIDER_FACES = ROOT / "data" / "unauthorized"
AUDIO = ROOT / "task3_audio"
OUTSIDER_VOICE = AUDIO / "unknown_yes_approve_1.wav"
MEMBERS = face_model.FINAL_MEMBERS
PHOTO_NAME = {"junior": "nkuba"}


def title(text):
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


def step(text):
    print("\n" + "-" * 70 + f"\n{text}\n" + "-" * 70)


def show(name):
    return name.replace("_", " ").title()


def wait(text):
    try:
        input(text)
    except EOFError:
        pass


def deny(stage, reason):
    print(f"\n  [ACCESS DENIED] {stage}: {reason}")
    print("  The flow stops here. No product recommendation is shown.")


def load_thresholds():
    face = joblib.load(MODELS_DIR / "face_model.joblib")["unknown_threshold"]
    voice = joblib.load(MODELS_DIR / "voice_model.joblib")["threshold"]
    return float(face), float(voice)


def run_transaction(face_path, voice_path=None, ask_voice=None, pause=False):
    """Run one attempt. Returns 'approved', 'denied at face' or 'denied at voice'."""
    face_limit, voice_limit = load_thresholds()

    step("STEP 1 of 3 - FACE RECOGNITION")
    print(f"  Image: {face_path}")
    try:
        face = face_model.predict_image(face_path)
    except Exception as error:
        deny("Face recognition", f"could not process the image ({error})")
        return "denied at face"
    print(f"  Face found in image     : {'yes' if face['face_detected'] else 'NO'}")
    print(f"  Closest registered user : {show(face['best_guess'])} (confidence {face['confidence']:.0%})")
    print(f"  Distance to known faces : {face['distance']:.2f} (must be <= {face_limit:.2f})")
    if not face["authorized"]:
        reason = "this face does not match any registered user" if face["face_detected"] \
            else "no face could be detected in this image"
        deny("Face recognition", reason)
        return "denied at face"
    user = face["user"]
    print(f"\n  [FACE OK] Recognised {show(user)}. A voice confirmation is needed to run the prediction.")
    if pause:
        wait("\n  Press Enter to request the prediction... ")

    step("STEP 2 of 3 - VOICE VERIFICATION")
    print('  Say "Yes, approve" or "Confirm transaction".')
    if voice_path is None and ask_voice:
        voice_path = ask_voice()
    if voice_path is None:
        deny("Voice verification", "no voice sample was provided")
        return "denied at voice"
    print(f"  Audio: {voice_path}")
    try:
        row, seconds = audio_pipeline.audio_file_to_feature_row(voice_path)
        voice = voice_model.predict_features(row, claimed_user=user)
    except Exception as error:
        deny("Voice verification", str(error))
        return "denied at voice"
    ranking = sorted(voice["probabilities"].items(), key=lambda item: -item[1])
    print(f"  Speech length            : {seconds:.2f} s")
    print(f"  Voice match for {show(user):<8} : {voice['confidence']:.2f} (must be >= {voice_limit:.2f})")
    print("  Voice sounds like        : " + ", ".join(f"{show(n)} {p:.2f}" for n, p in ranking))
    if not voice["authorized"]:
        hint = f" (it sounds more like {show(voice['best_guess'])})" if voice["best_guess"] != user else ""
        deny("Voice verification", f"the voice does not match {show(user)}{hint}")
        return "denied at voice"
    print("\n  [VOICE OK] Transaction approved.")

    step("STEP 3 of 3 - PRODUCT RECOMMENDATION")
    result = product_model.recommend(user, top_k=3)
    print(f"  Customer profile used: #{result['simulated_customer_id']} (simulated link for {show(user)})")
    print(f"\n  PREDICTED PRODUCT: {result['prediction']}\n")
    for item in result["top_k"]:
        print(f"    {item['product_category']:<12}{item['probability']:>5.0%}  " + "#" * round(item["probability"] * 30))
    print("\n  [TRANSACTION COMPLETE]")
    return "approved"


def face_sample(member):
    return IMAGES / f"{PHOTO_NAME.get(member, member)}_neutral.jpg"


def voice_sample(member, phrase="yes_approve"):
    return AUDIO / f"{member}_{phrase}_1.wav"


def outsider_face(path=None):
    if path:
        return Path(path), True
    photos = sorted(p for p in OUTSIDER_FACES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if photos:
        return photos[0], True
    blank = Path(tempfile.gettempdir()) / "stand_in_non_member.jpg"
    cv2.imwrite(str(blank), np.full((480, 480, 3), 190, np.uint8))
    return blank, False


def scenario_authorized(member, pause):
    title(f"SCENARIO 1: FULL TRANSACTION - registered user ({show(member)})")
    return run_transaction(face_sample(member), voice_sample(member), pause=pause)


def scenario_unknown_face(path, pause):
    face, real = outsider_face(path)
    title("SCENARIO 2: UNAUTHORIZED ATTEMPT - face of someone who is not registered")
    if not real:
        print("  Note: no photo found in data/unauthorized/, so a blank stand-in image is used.")
    return run_transaction(face, voice_sample("david"), pause=pause)


def scenario_unknown_voice(member, pause):
    title(f"SCENARIO 3: UNAUTHORIZED ATTEMPT - registered face ({show(member)}), outsider's voice")
    return run_transaction(face_sample(member), OUTSIDER_VOICE, pause=pause)


def run_all(member, outsider_path, pause):
    cases = [
        ("Full transaction (registered user)", "approved", lambda: scenario_authorized(member, pause)),
        ("Unauthorized face", "denied at face", lambda: scenario_unknown_face(outsider_path, pause)),
        ("Unauthorized voice", "denied at voice", lambda: scenario_unknown_voice(member, pause)),
    ]
    results = []
    for name, expected, run in cases:
        if pause and results:
            wait("\nPress Enter for the next scenario... ")
        results.append((name, expected, run()))
    title("SUMMARY")
    print(f"  {'Scenario':<38}{'Expected':<18}{'Actual':<18}Result")
    for name, expected, actual in results:
        print(f"  {name:<38}{expected:<18}{actual:<18}{'OK' if actual == expected else 'UNEXPECTED'}")


def ask_path(kind):
    if kind == "face":
        samples = sorted(IMAGES.glob("*.jpg")) + sorted(OUTSIDER_FACES.glob("*.*"))
    else:
        samples = (sorted(AUDIO.glob("*_yes_approve_1.wav")) + sorted(AUDIO.glob("*_confirm_transaction_1.wav")))
        samples = [s for s in samples if not s.name.startswith("unknown")] + [OUTSIDER_VOICE]
    samples = [s for s in samples if s.is_file()]
    label = "face image" if kind == "face" else "voice recording"
    print(f"\n  Sample {label}s:")
    for number, sample in enumerate(samples, 1):
        print(f"    {number:>2}. {sample.relative_to(ROOT)}")
    while True:
        try:
            text = input("\n  Enter a number or paste a path (q to cancel): ").strip().strip("\"'")
        except EOFError:
            return None
        if text.lower() in {"q", "quit"}:
            return None
        if text.isdigit() and 1 <= int(text) <= len(samples):
            return samples[int(text) - 1]
        for option in (Path(text).expanduser(), ROOT / text):
            if text and option.is_file():
                return option
        print("  File not found. Try again.")


def ask_member():
    print("\n  Registered members: " + ", ".join(show(m) for m in MEMBERS))
    while True:
        try:
            name = unify_member(input("  Which member? [david]: ").strip() or "david")
        except EOFError:
            return "david"
        if name in MEMBERS:
            return name
        print("  Not a registered member.")


MENU = """
  1) Make a transaction (choose your own face image and voice clip)
  2) Demo: full transaction with a registered member
  3) Demo: unauthorized attempt - face of a non-member
  4) Demo: unauthorized attempt - registered face + outsider's voice
  5) Demo: run all three scenarios
  q) Quit
"""


def menu(outsider_path):
    title("MULTIMODAL AUTHENTICATION + PRODUCT RECOMMENDATION SYSTEM")
    while True:
        print(MENU)
        try:
            choice = input("  Choose an option: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in {"q", "quit"}:
            break
        if choice == "1":
            face = ask_path("face")
            if face:
                run_transaction(face, ask_voice=lambda: ask_path("voice"))
        elif choice == "2":
            scenario_authorized(ask_member(), pause=True)
        elif choice == "3":
            scenario_unknown_face(outsider_path, pause=True)
        elif choice == "4":
            scenario_unknown_voice(ask_member(), pause=True)
        elif choice == "5":
            run_all(ask_member(), outsider_path, pause=True)
        else:
            print("  Please choose one of the options in the menu.")
    print("\n  Goodbye.")


def main():
    parser = argparse.ArgumentParser(description="Task 6: multimodal authentication demo")
    parser.add_argument("--face", help="face image for a single transaction")
    parser.add_argument("--voice", help="voice recording for a single transaction")
    parser.add_argument("--demo", action="store_true", help="run the full transaction and both unauthorized attempts")
    parser.add_argument("--member", default="david", help="registered member used in the demos")
    parser.add_argument("--unknown-face", help="photo of a non-member for the unauthorized-face scenario")
    parser.add_argument("--no-pause", action="store_true", help="do not wait for Enter between steps")
    args = parser.parse_args()

    missing = [n for n in ("face_model.joblib", "voice_model.joblib", "product_model.joblib",
                           "member_to_customer.json") if not (MODELS_DIR / n).exists()]
    if missing:
        print(f"Missing files in {MODELS_DIR}: {', '.join(missing)}\nTrain them with: python src/train_all_models.py")
        return 2
    member = unify_member(args.member)
    if member not in MEMBERS:
        print(f"Unknown member '{args.member}'. Choose from: {', '.join(MEMBERS)}")
        return 2

    if args.face or args.voice:
        if not (args.face and args.voice):
            print("Give both --face and --voice.")
            return 2
        title("SINGLE TRANSACTION")
        return 0 if run_transaction(args.face, args.voice) == "approved" else 1
    if args.demo:
        run_all(member, args.unknown_face, pause=not args.no_pause)
    else:
        menu(args.unknown_face)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)