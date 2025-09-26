import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import webbrowser

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


def startapplication():
    video = cv2.VideoCapture('cars.mp4')  # for camera use video = cv2.VideoCapture(0)
    accident_reported = False  # flag to open HTML only once per accident

    # Path to your receiver.html
    file_path = r"C:\Users\lokeshwaran\OneDrive\Desktop\Accident-Detection-System-main\receiver.html"

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            # Draw prediction on video
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)

            # Open HTML with live geolocation only once
            if not accident_reported:
                html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Accident Detection Alert</title>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Roboto', sans-serif; background: linear-gradient(135deg, #f44336, #ff7961); color: #fff; text-align: center; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    h1 { font-size: 2.5rem; margin-bottom: 10px; }
    #alert { background-color: #ffeb3b; color: #f44336; font-size: 1.8rem; font-weight: bold; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
    #status { margin-top: 15px; font-size: 1.2rem; }
    #map { width: 90%; max-width: 600px; height: 400px; margin-top: 20px; border: 4px solid #fff; border-radius: 15px; }
</style>
</head>
<body>
<h1>Real-Time Accident Detection</h1>
<div id="alert">Accident Detected!</div>
<p id="status">Locating...</p>
<div id="map"></div>

<script>
    if (navigator.geolocation) {
        navigator.geolocation.watchPosition(showPosition, showError, { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 });
    } else {
        document.getElementById('status').innerHTML = "Geolocation is not supported by this browser.";
    }

    function showPosition(position) {
        const lat = position.coords.latitude.toFixed(6);
        const lon = position.coords.longitude.toFixed(6);
        document.getElementById('status').innerHTML = `Latitude: ${lat} <br> Longitude: ${lon} <br> Accuracy: ${position.coords.accuracy} meters`;
        document.getElementById('map').innerHTML = `<iframe width="100%" height="100%" src="https://www.google.com/maps?q=${lat},${lon}&hl=es;z=18&output=embed"></iframe>`;
    }

    function showError(error) {
        const status = document.getElementById('status');
        switch(error.code) {
            case error.PERMISSION_DENIED: status.innerHTML = "User denied the request for Geolocation."; break;
            case error.POSITION_UNAVAILABLE: status.innerHTML = "Location information is unavailable."; break;
            case error.TIMEOUT: status.innerHTML = "The request to get user location timed out."; break;
            case error.UNKNOWN_ERROR: status.innerHTML = "An unknown error occurred."; break;
        }
    }
</script>
</body>
</html>
"""
                with open(file_path, "w") as f:
                    f.write(html_content)

                # Open in default browser
                webbrowser.open('file://' + file_path)
                accident_reported = True

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        cv2.imshow('Video', frame)

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    startapplication()
