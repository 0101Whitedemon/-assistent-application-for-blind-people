"""
VisionGuide AI
Computer Vision + AI voice assistant for navigation support.

Run in VS Code:
    python main.py

Safety:
This is an assistive prototype. It can miss objects and cannot reliably measure
distance, road safety, stairs, drop-offs, or traffic. Do not use it as the only
navigation aid. A white cane/guide dog and normal safety practices remain important.
"""

import cv2
import time
import threading
import queue
import re
import sys

from ultralytics import YOLO

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


MODEL_FILE = "yolo11n.pt"
CONFIDENCE = 0.45
SPEAK_DELAY = 2.5
COMMAND_COOLDOWN = 1.5

# Objects that deserve a stronger warning when ahead.
HIGH_RISK = {
    "car", "truck", "bus", "motorcycle", "bicycle", "train"
}

speech_queue = queue.Queue(maxsize=2)
running = True

warnings_enabled = True
last_spoken = ""
last_spoken_time = 0.0
last_command_time = 0.0

current_detections = []
current_warnings = []
current_frame = None
state_lock = threading.Lock()


# =========================================================
# VOICE OUTPUT
# =========================================================

def voice_worker():
    """Runs text-to-speech in a background thread."""
    if pyttsx3 is None:
        print("[VOICE] pyttsx3 not installed. Voice output disabled.")
        while running:
            try:
                msg = speech_queue.get(timeout=0.5)
                if msg is None:
                    break
                print("[VOICE]", msg)
                speech_queue.task_done()
            except queue.Empty:
                pass
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)

        while running:
            try:
                msg = speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if msg is None:
                speech_queue.task_done()
                break

            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception as exc:
                print("[VOICE ERROR]", exc)

            speech_queue.task_done()

        engine.stop()
    except Exception as exc:
        print("[VOICE INIT ERROR]", exc)


def speak(message, force=False):
    """Speak the newest message without building a huge backlog."""
    global last_spoken, last_spoken_time

    message = re.sub(r"\s+", " ", str(message)).strip()
    if not message:
        return

    now = time.time()

    if not force:
        if message == last_spoken and now - last_spoken_time < SPEAK_DELAY:
            return

    last_spoken = message
    last_spoken_time = now

    # Remove old queued speech.
    try:
        while True:
            speech_queue.get_nowait()
            speech_queue.task_done()
    except queue.Empty:
        pass

    try:
        speech_queue.put_nowait(message)
    except queue.Full:
        pass


# =========================================================
# CAMERA
# =========================================================

def open_camera():
    """Find the first working webcam on Windows."""
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    for camera_index in range(5):
        for backend in backends:
            camera = cv2.VideoCapture(camera_index, backend)

            if not camera.isOpened():
                camera.release()
                continue

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            ok, frame = camera.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[CAMERA] Camera {camera_index} opened.")
                return camera

            camera.release()

    return None


# =========================================================
# COMPUTER VISION
# =========================================================

def get_position(center_x, frame_width):
    if center_x < frame_width * 0.34:
        return "left"
    if center_x > frame_width * 0.66:
        return "right"
    return "ahead"


def estimate_distance(box, frame_width, frame_height):
    """
    Approximate distance using bounding-box size.
    This is NOT real-world distance measurement.
    """
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    area_ratio = (bw * bh) / float(frame_width * frame_height)
    width_ratio = bw / float(frame_width)

    if area_ratio > 0.30 or width_ratio > 0.60:
        return "very close"
    if area_ratio > 0.14 or width_ratio > 0.38:
        return "close"
    if area_ratio > 0.05 or width_ratio > 0.20:
        return "nearby"
    return "far"


def warning_for(name, position, distance):
    if name in HIGH_RISK and position == "ahead":
        return (
            f"Warning. {name} ahead. "
            "Please stop and check your surroundings."
        )

    if name == "train":
        return (
            f"Warning. Train detected on the {position}. "
            "Stay away and check your surroundings."
        )

    if name == "traffic light" and position == "ahead":
        return (
            "Traffic light ahead. Do not cross based on this detection alone. "
            "Check the signal and traffic."
        )

    if name == "stop sign" and position == "ahead":
        return "Stop sign ahead. Slow down and check for traffic."

    if position == "ahead" and distance == "very close":
        return f"Stop. {name} is very close ahead."

    if position == "ahead" and distance == "close":
        return f"Caution. {name} is close ahead."

    if distance == "very close":
        return f"Obstacle very close on the {position}. Move carefully."

    return None


def priority(item):
    name = item["name"]
    position = item["position"]
    distance = item["distance"]

    if name in HIGH_RISK and position == "ahead":
        return 100
    if name == "train":
        return 95
    if name == "traffic light" and position == "ahead":
        return 80
    if name == "stop sign" and position == "ahead":
        return 75
    if position == "ahead" and distance == "very close":
        return 70
    if position == "ahead" and distance == "close":
        return 60
    if distance == "very close":
        return 50
    return 0


def analyze_frame(model, frame):
    """YOLO object detection + position + rough distance + warning."""
    height, width = frame.shape[:2]
    detections = []
    warnings = []

    results = model(
        frame,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE:
                continue

            name = str(model.names[class_id])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            position = get_position(center_x, width)
            distance = estimate_distance(
                (x1, y1, x2, y2),
                width,
                height
            )

            warning = warning_for(name, position, distance)

            item = {
                "name": name,
                "confidence": confidence,
                "position": position,
                "distance": distance,
                "box": (x1, y1, x2, y2),
                "warning": warning,
            }

            detections.append(item)

            if warning:
                warnings.append(item)

            draw_color = (0, 0, 255) if warning else (0, 200, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                draw_color,
                2
            )

            cv2.putText(
                frame,
                f"{name} {confidence:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2
            )

            cv2.putText(
                frame,
                f"{position}, {distance}",
                (x1, min(height - 10, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2
            )

    warnings.sort(key=priority, reverse=True)
    detections.sort(key=priority, reverse=True)

    return frame, detections, warnings


def scene_summary(detections):
    if not detections:
        return "I do not detect a known object in the current view."

    counts = {}
    for item in detections:
        counts[item["name"]] = counts.get(item["name"], 0) + 1

    parts = []
    for name, count in counts.items():
        if count == 1:
            pos = next(
                d["position"] for d in detections
                if d["name"] == name
            )
            if pos == "ahead":
                parts.append(f"{name} ahead")
            else:
                parts.append(f"{name} on the {pos}")
        else:
            parts.append(f"{count} {name}s")

    return "I detect " + ", ".join(parts) + "."


# =========================================================
# OCR: READ TEXT
# =========================================================

def read_text(frame):
    if pytesseract is None:
        return (
            "Text reading is disabled. "
            "Install pytesseract and Tesseract OCR to enable it."
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6)

    # Improve contrast for ordinary printed text.
    processed = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    try:
        text = pytesseract.image_to_string(processed).strip()
    except Exception as exc:
        return f"Text reading failed: {exc}"

    if not text:
        return "I cannot read clear text in the current view."

    text = re.sub(r"\s+", " ", text)
    return "I can read: " + text[:600]


# =========================================================
# VOICE INPUT / COMMANDS
# =========================================================

def listen_for_command():
    """
    Optional microphone input.
    speech_recognition's Google recognizer requires internet.
    If unavailable, keyboard commands still work.
    """
    if sr is None:
        speak(
            "Voice commands are not installed. "
            "Use the keyboard controls.",
            force=True
        )
        return ""

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.7
    recognizer.energy_threshold = 300

    try:
        with sr.Microphone() as source:
            speak("Listening.", force=True)
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source,
                timeout=VOICE_LISTEN_SECONDS,
                phrase_time_limit=VOICE_LISTEN_SECONDS
            )

        try:
            command = recognizer.recognize_google(audio)
            command = command.lower().strip()
            print("[COMMAND]", command)
            return command
        except sr.UnknownValueError:
            speak("I could not understand that.", force=True)
        except sr.RequestError:
            speak(
                "Voice recognition needs an internet connection. "
                "Use the keyboard controls.",
                force=True
            )

    except Exception as exc:
        print("[MIC ERROR]", exc)
        speak("Microphone is unavailable.", force=True)

    return ""


def process_command(command, frame):
    global warnings_enabled

    if not command:
        return

    command = command.lower()

    if "what" in command and (
        "front" in command or
        "ahead" in command or
        "there" in command
    ):
        with state_lock:
            detections = list(current_detections)
        speak(scene_summary(detections), force=True)
        return

    if "describe" in command or "scene" in command:
        with state_lock:
            detections = list(current_detections)
        speak(scene_summary(detections), force=True)
        return

    if "read" in command or "text" in command:
        speak("Reading text.", force=True)
        result = read_text(frame)
        speak(result, force=True)
        return

    if "repeat" in command:
        if last_spoken:
            speak(last_spoken, force=True)
        else:
            speak("There is nothing to repeat.", force=True)
        return

    if "stop warning" in command or "disable warning" in command:
        warnings_enabled = False
        speak("Warnings disabled.", force=True)
        return

    if "start warning" in command or "enable warning" in command:
        warnings_enabled = True
        speak("Warnings enabled.", force=True)
        return

    if "help" in command or "what can you do" in command:
        speak(
            "You can ask what is in front, describe the scene, "
            "read text, repeat the last message, or enable and disable warnings.",
            force=True
        )
        return

    speak(
        "I did not recognize that command. Say help for available commands.",
        force=True
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():
    global running, current_frame, current_detections, current_warnings

    print("=" * 65)
    print("VISIONGUIDE AI")
    print("Computer Vision + Voice Assistant")
    print("=" * 65)

    if not os_path_exists(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} was not found.")
        print("Put yolo11n.pt in the same folder as main.py.")
        return

    print("[AI] Loading YOLO model...")
    model = YOLO(MODEL_FILE)
    print("[AI] Model loaded.")

    camera = open_camera()

    if camera is None:
        print("ERROR: No working camera found.")
        print("Check Windows camera permissions.")
        return

    threading.Thread(
        target=voice_worker,
        daemon=True
    ).start()

    speak(
        "Vision Guide started. Camera is scanning. "
        "Press H for help.",
        force=True
    )

    previous_warning_key = ""
    previous_warning_time = 0.0

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                speak("Camera error.", force=True)
                break

            processed, detections, warnings = analyze_frame(
                model,
                frame
            )

            with state_lock:
                current_frame = frame.copy()
                current_detections = detections
                current_warnings = warnings

            # Speak only the highest-priority warning.
            if warnings_enabled and warnings:
                top = warnings[0]
                warning = top["warning"]

                if warning:
                    warning_key = (
                        top["name"],
                        top["position"],
                        top["distance"]
                    )
                    now = time.time()

                    if (
                        warning_key != previous_warning_key
                        or now - previous_warning_time > SPEAK_DELAY
                    ):
                        speak(warning)
                        previous_warning_key = warning_key
                        previous_warning_time = now

            # UI overlay.
            cv2.putText(
                processed,
                "VISIONGUIDE AI",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                processed,
                "Q Quit | H Help | V Voice | R Read Text | SPACE Repeat",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            status = (
                "WARNINGS ON"
                if warnings_enabled
                else "WARNINGS OFF"
            )

            cv2.putText(
                processed,
                status,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.imshow("VisionGuide AI", processed)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("h"):
                speak(
                    "Controls: V for voice command, R to read text, "
                    "Space to repeat, W to toggle warnings, and Q to quit.",
                    force=True
                )

            elif key == ord("v"):
                command = listen_for_command()
                process_command(command, frame)

            elif key == ord("r"):
                speak("Reading text.", force=True)
                speak(read_text(frame), force=True)

            elif key == ord("w"):
                warnings_enabled = not warnings_enabled
                speak(
                    "Warnings enabled."
                    if warnings_enabled
                    else "Warnings disabled.",
                    force=True
                )

            elif key == 32:  # SPACE
                if last_spoken:
                    speak(last_spoken, force=True)

    finally:
        running = False
        camera.release()
        cv2.destroyAllWindows()

        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass

        print("VisionGuide stopped.")


def os_path_exists(path):
    # Small helper keeps the project dependency-free.
    import os
    return os.path.exists(path)


if __name__ == "__main__":
    main()
