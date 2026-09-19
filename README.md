# Cat Meme Gesture Detector

Your webcam reacts to your face and hands with memes. Raise your eyebrows, open your mouth, put a finger on your lips, or make a shocked face, and the matching meme pops up next to your live camera feed, in real time.


## Gestures

| Gesture | Meme |
|---|---|
| Both hands on either side of the face | `cara.jpeg` |
| Both hands above the nose | `sonic.png` |
| Index finger on the mouth (shhh) | `cristiano.png` |
| Peace sign (index + middle finger up) | `rata.jpeg` |
| Mouth wide open + eyebrows raised (shock) | `cat.png` |
| Mouth open, eyebrows normal | `gato.png` |
| Eyebrows raised | `perro.png` |

If two gestures match at the same time, the one higher in this table wins (the order is set by `PRIORITY` in the code).

## How it works

No model is trained. Pre-trained MediaPipe models find landmarks on your face and hands, and simple geometry rules on those points decide which gesture you are making.

1. **Capture:** OpenCV reads the webcam and mirrors the frame like a selfie.
2. **Landmarks:** MediaPipe Face Mesh finds facial points and MediaPipe Hands finds 21 points per hand.
3. **Calibration:** for the first ~45 frames it records your neutral face (eyebrow height and mouth opening) so the thresholds fit *your* face.
4. **Rules:** distances between landmarks are divided by face height, so results don't change if you sit closer to or farther from the camera. For example, "eyebrows raised" means the eyebrow-to-eyelid gap is bigger than your neutral gap.
5. **Smoothing:** the meme only changes when a gesture wins at least 5 of the last 8 frames, which stops flickering.

## Tech stack

- Python
- OpenCV (camera, drawing, windows)
- MediaPipe Face Mesh + Hands
- NumPy

## Setup

Requires **Python 3.8 to 3.11** (MediaPipe does not support newer versions).

```bash
git clone https://github.com/dhairya3343/CAT_MEMES.git
cd CAT_MEMES

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python3 my_meme_.py             # Windows: python my_meme_.py
```

**macOS:** the first time, allow camera access for your terminal / VS Code when asked (System Settings → Privacy & Security → Camera), then run it again.

## Controls

| Key | Action |
|---|---|
| `ESC` | Quit |
| `C` | Recalibrate your neutral face |
| `D` | Show / hide debug numbers |

Look straight at the camera with a neutral face while it calibrates. Good lighting helps a lot.

## Tuning and adding your own memes

- **Tuning:** the debug numbers on screen (`brow_ratio`, `mouth_extra`, `finger_dist`) can be compared with the values in the `T = {...}` dictionary at the top of `my_meme_.py`. If a gesture triggers too easily or too rarely, adjust that value.
- **New meme:** put the image in the project folder, add a line to the `MEMES` dictionary and the gesture name to `PRIORITY`. New gesture rules go in the `detect()` function.

## Known limitations

- Face Mesh does not track the tongue, so "mouth open with normal eyebrows" is used instead of a real tongue-out detector.
- Finger detection assumes your hand is roughly upright.
- Thresholds are starting values and may need tuning for your face and lighting.

## Credits

- Inspired by (https://github.com/jayesh-cmd/cat_meme_gesture.git) by Jayesh Vishwakarma
jayesh-cmd.
- Written with AI assistance (Claude); gestures, memes and tuning were tested and adjusted by me.
- Meme images belong to their respective owners and are used here for learning purposes only. Replace them with your own if you plan to reuse or share the project widely.
