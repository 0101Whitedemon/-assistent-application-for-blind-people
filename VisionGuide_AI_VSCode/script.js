let cameraStream = null;

const video = document.getElementById("camera");
const objectText = document.getElementById("object");
const directionText = document.getElementById("direction");
const warningText = document.getElementById("warning");

async function startCamera() {
    if (cameraStream) {
        warningText.textContent = "Status: Camera already running";
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
        warningText.textContent =
            "Camera unavailable. Use HTTPS or localhost.";
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        objectText.textContent = "Camera is working";
        directionText.textContent = "Direction: Scanning";
        warningText.textContent = "Status: Camera preview only";

        speak("Vision Guide camera started.");
    } catch (error) {
        console.error("Camera error:", error);
        warningText.textContent =
            "Allow camera access and try again.";
    }
}

function stopCamera() {
    if (!cameraStream) {
        warningText.textContent = "Status: Camera already stopped";
        return;
    }

    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    video.srcObject = null;

    objectText.textContent = "Camera stopped";
    directionText.textContent = "Direction: -";
    warningText.textContent = "Status: Offline";

    speak("Vision Guide camera stopped.");
}

function speak(message) {
    if (!("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(message);
    speech.rate = 0.9;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);
}
