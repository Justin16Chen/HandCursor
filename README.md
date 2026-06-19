# HandCursor

A real-time hand tracking system that will eventually control the mouse with
your hand. Right now it opens your webcam, detects hand landmarks with
MediaPipe, prints their coordinates, and shows a live preview.

## Features

- Webcam capture with OpenCV (`camera.py`)
- Hand landmark detection with MediaPipe (`hand_tracking.py`)
- Mouse control scaffolding with PyAutoGUI (`mouse_controller.py`, not wired up yet)
- Live preview with drawn landmarks and an FPS counter (`main.py`)

## Project structure

```
HandCursor/
├── main.py              # Entry point
├── camera.py            # Webcam capture
├── hand_tracking.py     # MediaPipe hand landmark detection
├── mouse_controller.py  # PyAutoGUI mouse control (placeholder)
├── requirements.txt
└── README.md
```

## Requirements

> [!IMPORTANT]
> MediaPipe currently supports **Python 3.9–3.12**. If you are on a newer
> version (e.g. 3.14), create a virtual environment with a supported Python
> before installing dependencies, otherwise `pip install mediapipe` will fail.

## Setup

```bash
# (Recommended) create a virtual environment with Python 3.12
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press `q` in the preview window to quit.

## Landmark reference

MediaPipe returns 21 landmarks per hand (IDs `0`–`20`). For example, ID `8`
is the index fingertip and ID `4` is the thumb tip. These will be used later
to map hand position to cursor movement and clicks.
