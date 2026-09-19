"""
Gesture Meme - your webcam shows a meme that matches your face / hand gesture.

Keys:  ESC = quit   C = recalibrate   D = show/hide debug numbers

Needs:  pip install opencv-python "mediapipe==0.10.14" "numpy<2"
"""
import os
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np


# 1. SETTINGS - this is the only part you need to edit to add memes


# gesture name -> image file (must be in the same folder as this script)
MEMES = {
       "hands_sides":  "cara.jpeg",
       "hands_up":     "sonic.png",
       "finger_mouth": "cristiano.png",
       "peace":        "rata.jpeg",
       "shock":        "cat.png",
       "mouth_open":   "gato.png",
       "eyebrows":     "perro.png",
   }

# If two gestures match at the same time, the FIRST one in this list wins.
PRIORITY = ["hands_sides", "hands_up", "finger_mouth", "peace",
            "shock", "mouth_open", "eyebrows"]

# Tuning values. Start here if a gesture triggers too easily / too rarely.
T = {
    "brow_raise":  1.12,  # brow-to-eye gap must be 12% bigger than your neutral face
    "mouth_open":  0.05,  # extra mouth opening (as fraction of face height)
    "mouth_wide":  0.10,  # mouth opening needed for "shock"
    "finger_mouth": 0.25, # fingertip-to-mouth distance (fraction of face height)
}

CALIB_FRAMES = 45   # frames used to learn your neutral face
VOTE_WINDOW  = 8    # look at the last 8 frames...
MIN_VOTES    = 5    # ...a gesture must win at least 5 of them (stops flicker)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils



# 2. GEOMETRY HELPERS

def pt(lm, i, W, H):
    """Landmark i as a pixel position (numpy array)."""
    return np.array([lm[i].x * W, lm[i].y * H])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def face_metrics(face, W, H):
    """
    Numbers that describe the face, all divided by face height so they do
    not change when you sit closer to / farther from the camera.
      brow  = gap between eyebrow and eyelid (bigger when brows go up)
      mouth = gap between inner lips        (bigger when mouth opens)
    """
    scale = dist(pt(face, 10, W, H), pt(face, 152, W, H)) + 1e-6
    brow_l = dist(pt(face, 105, W, H), pt(face, 159, W, H))
    brow_r = dist(pt(face, 334, W, H), pt(face, 386, W, H))
    mouth  = dist(pt(face, 13, W, H), pt(face, 14, W, H))
    return {
        "scale": scale,
        "brow":  (brow_l + brow_r) / 2 / scale,
        "mouth": mouth / scale,
    }


def fingers_up(hand):
    """[index, middle, ring, pinky] -> True if that finger is straight up."""
    tips = (8, 12, 16, 20)
    pips = (6, 10, 14, 18)
    return [hand[t].y < hand[p].y for t, p in zip(tips, pips)]



# 3. GESTURE DETECTION

def detect(face, hands, W, H, base):
    """
    Returns (found, debug).
    found = {gesture_name: True/False}, debug = numbers shown on screen.
    """
    found = {name: False for name in PRIORITY}
    debug = {}

    # ---- hand-only gestures ----
    states = [fingers_up(h) for h in hands]
    debug["fingers"] = states

    for s in states:
        if s == [True, True, False, False]:
            found["peace"] = True

    # ---- gestures that need the face ----
    if face:
        m = face_metrics(face, W, H)
        brows_up   = m["brow"] > base["brow"] * T["brow_raise"]
        extra      = m["mouth"] - base["mouth"]
        mouth_open = extra > T["mouth_open"]
        mouth_wide = extra > T["mouth_wide"]

        found["eyebrows"]   = brows_up
        found["mouth_open"] = mouth_open and not brows_up
        found["shock"]      = mouth_wide and brows_up

        debug["brow_ratio"] = m["brow"] / base["brow"]   # want > brow_raise
        debug["mouth_extra"] = extra                     # want > mouth_open

        # index finger on mouth: index up, middle finger folded, tip near lips
        mouth_pt = pt(face, 13, W, H)
        for hand, s in zip(hands, states):
            if s[0] and not s[1]:
                near = dist(pt(hand, 8, W, H), mouth_pt) / m["scale"]
                debug["finger_dist"] = near
                if near < T["finger_mouth"]:
                    found["finger_mouth"] = True

        # two-hand gestures
        if len(hands) == 2:
            nose_y = face[1].y
            found["hands_up"] = all(h[9].y < nose_y for h in hands)

            x_left, x_right = sorted([face[234].x, face[454].x])
            y_top, y_bottom = face[10].y, face[152].y
            hx = sorted(h[9].x for h in hands)
            level = all(y_top < h[9].y < y_bottom for h in hands)
            found["hands_sides"] = hx[0] < x_left and hx[1] > x_right and level

    return found, debug



# 4. DRAWING / DISPLAY HELPERS

def load_memes():
    memes = {}
    for gesture, filename in MEMES.items():
        img = cv2.imread(os.path.join(BASE_DIR, filename))
        if img is None:
            print(f"[warning] could not load '{filename}' for '{gesture}'")
        memes[gesture] = img
    return memes


def meme_panel(memes, cache, name, W, H):
    """Image to show in the second window (resized once, then cached)."""
    key = (name, W, H)
    if key in cache:
        return cache[key]
    panel = np.full((H, W, 3), 30, dtype=np.uint8)
    if name is None:
        cv2.putText(panel, "neutral", (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (160, 160, 160), 2, cv2.LINE_AA)
    elif memes.get(name) is None:
        cv2.putText(panel, f"missing: {MEMES[name]}", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 220), 2, cv2.LINE_AA)
    else:
        panel = cv2.resize(memes[name], (W, H))
    cache[key] = panel
    return panel


def draw_text(frame, text, pos, color=(230, 230, 230), scale=0.6):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)          # dark outline
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_calibration(frame, progress):
    H, W = frame.shape[:2]
    dark = frame.copy()
    cv2.rectangle(dark, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(dark, 0.55, frame, 0.45, 0, frame)
    draw_text(frame, "Look straight - keep a neutral face",
              (W // 2 - 190, H // 2 - 25), scale=0.8)
    x1, x2, y = W // 2 - 140, W // 2 + 140, H // 2 + 10
    cv2.rectangle(frame, (x1, y), (x2, y + 18), (60, 60, 60), -1)
    cv2.rectangle(frame, (x1, y), (x1 + int(280 * progress), y + 18), (80, 220, 80), -1)


def draw_hud(frame, shown, debug, show_debug):
    draw_text(frame, shown if shown else "neutral", (12, 30),
              (80, 230, 80) if shown else (170, 170, 170), 0.9)
    if show_debug:
        y = 58
        for key in ("brow_ratio", "mouth_extra", "finger_dist"):
            if key in debug:
                draw_text(frame, f"{key}: {debug[key]:.2f}", (12, y))
                y += 22
        for i, s in enumerate(debug.get("fingers", [])):
            draw_text(frame, f"hand {i + 1} fingers: {[int(v) for v in s]}", (12, y))
            y += 22


def open_camera():
    for index in (0, 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(5):          # let the camera settle
                cap.read()
            return cap
        cap.release()
    return None



# 5. MAIN LOOP

def main():
    cap = open_camera()
    if cap is None:
        print("Could not open the camera. Check camera permission and close other apps using it.")
        return

    memes = load_memes()
    cache = {}
    face_mesh = mp_face.FaceMesh(max_num_faces=1,
                                 min_detection_confidence=0.6,
                                 min_tracking_confidence=0.6)
    hands_model = mp_hands.Hands(max_num_hands=2,
                                 min_detection_confidence=0.6,
                                 min_tracking_confidence=0.6)

    base = None            # neutral-face numbers, filled in by calibration
    samples = []
    votes = deque(maxlen=VOTE_WINDOW)
    shown = None
    show_debug = True

    cv2.namedWindow("Your Camera")
    cv2.namedWindow("Meme")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)          # mirror, like a selfie
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_res = face_mesh.process(rgb)
        hand_res = hands_model.process(rgb)

        face = (face_res.multi_face_landmarks[0].landmark
                if face_res.multi_face_landmarks else None)
        hand_lms = hand_res.multi_hand_landmarks or []
        hands = [h.landmark for h in hand_lms]
        for h in hand_lms:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        debug = {}
        if base is None:
            # ---- calibration: learn the neutral face ----
            if face:
                samples.append(face_metrics(face, W, H))
            draw_calibration(frame, min(len(samples) / CALIB_FRAMES, 1.0))
            if len(samples) >= CALIB_FRAMES:
                base = {
                    "brow":  float(np.median([s["brow"] for s in samples])),
                    "mouth": float(np.median([s["mouth"] for s in samples])),
                }
                print("Calibrated:", base)
            meme_img = meme_panel(memes, cache, None, W, H)
        else:
            found, debug = detect(face, hands, W, H, base)
            current = next((g for g in PRIORITY if found[g]), None)

            votes.append(current)
            winner, count = Counter(votes).most_common(1)[0]
            if count >= MIN_VOTES:
                shown = winner

            draw_hud(frame, shown, debug, show_debug)
            meme_img = meme_panel(memes, cache, shown, W, H)

        cv2.imshow("Your Camera", frame)
        cv2.imshow("Meme", meme_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC
            break
        if key == ord("c"):                 # recalibrate
            base, samples, shown = None, [], None
            votes.clear()
        if key == ord("d"):
            show_debug = not show_debug

    face_mesh.close()
    hands_model.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()