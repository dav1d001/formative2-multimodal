"""Train all Task 4 models in a reproducible, readable order."""

try:
    from .face_model import train as train_face
    from .voice_model import train as train_voice
    from .product_model import train as train_product
except ImportError:
    from face_model import train as train_face
    from voice_model import train as train_voice
    from product_model import train as train_product


def main():
    print("\n" + "=" * 72 + "\nTASK 4.1: FACE RECOGNITION\n" + "=" * 72)
    train_face()
    print("\n" + "=" * 72 + "\nTASK 4.2: VOICE IDENTIFICATION AND VERIFICATION\n" + "=" * 72)
    train_voice()
    print("\n" + "=" * 72 + "\nTASK 4.3: PRODUCT RECOMMENDATION\n" + "=" * 72)
    train_product()


if __name__ == "__main__":
    main()