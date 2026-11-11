"""Smoke-test helper ONLY: draws cartoon placeholder 'faces' so the pipeline can be run end-to-end
before real photos exist. NEVER submit these - replace with real photos of each team member."""
import sys, numpy as np, cv2
from pathlib import Path
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/images_demo"); out.mkdir(parents=True, exist_ok=True)
members = {"demo_a": ((190, 160, 140), (40, 30, 20), (230, 225, 215)),
           "demo_b": ((120, 85, 65), (15, 15, 15), (200, 215, 230)),
           "demo_c": ((235, 205, 185), (120, 80, 40), (215, 230, 215))}
for name, (skin, hair, bg) in members.items():
    for expr in ["neutral", "smiling", "surprised"]:
        im = np.full((480, 480, 3), bg[::-1], np.uint8); c = (240, 250)
        cv2.ellipse(im, (240, 190), (175, 120), 0, 180, 360, hair[::-1], -1)
        cv2.ellipse(im, c, (130, 165), 0, 0, 360, skin[::-1], -1)
        er = 22 if expr == "surprised" else 14
        for dx in (-50, 50):
            cv2.circle(im, (240 + dx, 225), er, (255, 255, 255), -1); cv2.circle(im, (240 + dx, 225), er // 2, (30, 20, 10), -1)
            by = 180 if expr == "surprised" else 195
            cv2.line(im, (240 + dx - 25, by), (240 + dx + 25, by), hair[::-1], 6)
        cv2.line(im, (240, 240), (240, 285), tuple(int(v * .8) for v in skin[::-1]), 3)
        if expr == "neutral": cv2.line(im, (200, 330), (280, 330), (60, 40, 120), 5)
        elif expr == "smiling": cv2.ellipse(im, (240, 315), (50, 30), 0, 10, 170, (60, 40, 120), 6)
        else: cv2.ellipse(im, (240, 335), (24, 34), 0, 0, 360, (40, 20, 70), -1)
        cv2.imwrite(str(out / f"{name}_{expr}.jpg"), im)
print("wrote placeholders to", out)
