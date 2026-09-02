# VisionGuide AI

A VS Code / Python computer-vision prototype that uses YOLO object detection,
OpenCV, text-to-speech, optional speech recognition, and OCR.

## Features

- Live webcam object detection with YOLO11
- Left / ahead / right position estimation
- Rough "far / nearby / close / very close" estimate from image size
- Priority safety warnings
- AI voice output
- Optional voice commands
- Read visible printed text with OCR
- Scene description on request
- Repeat the last spoken message
- Enable/disable warnings
- Keyboard fallback if microphone is unavailable

## Project files

- `main.py` -> complete application
- `yolo11n.pt` -> YOLO model
- `requirements.txt` -> Python packages
- `README.md` -> setup instructions
- `style.css` and `script.js` -> optional web UI files from the earlier prototype

## VS Code setup

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:

   Windows:
   `python -m venv .venv`

5. Activate it:

   PowerShell:
   `.\.venv\Scripts\Activate.ps1`

   If PowerShell blocks activation, use:
   `.\.venv\Scripts\activate.bat`

6. Install packages:

   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`

7. Run:

   `python main.py`

## If PyAudio fails on Windows

Try:

`python -m pip install pipwin`

Then:

`pipwin install pyaudio`

If microphone support still fails, the application will still work with
keyboard controls and voice output if pyttsx3 is installed.

## OCR setup

`pytesseract` is a Python wrapper. Tesseract OCR itself must also be installed
on Windows. If Tesseract is not in PATH, add its installation folder to PATH.

OCR is optional. Object detection and voice output do not depend on OCR.

## Controls

- `V` = listen for a voice command
- `R` = read text from the current camera view
- `SPACE` = repeat last voice message
- `W` = enable/disable warnings
- `H` = voice help
- `Q` = quit

### Example voice commands

- "What is in front of me?"
- "Describe the scene"
- "Read the text"
- "Repeat"
- "Stop warnings"
- "Start warnings"
- "Help"

## Important safety limitation

The distance calculation is only a rough visual estimate. A normal webcam
does not know exact physical distance without calibration/depth sensing.

The detector may miss objects, misunderstand scenes, or produce false
detections. It does not reliably detect every hazard such as stairs, holes,
curbs, glass, low obstacles, road edges, or moving traffic.

For a real assistive product, add depth sensing, GPS/navigation, better
segmentation, obstacle/ground detection, fall/drop-off detection, multilingual
speech, an emergency contact feature, and extensive real-world testing.

Do not rely on this prototype alone for navigation.
