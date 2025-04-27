import cv2
from ultralytics import YOLO
import threading
import time
from flask import Flask, render_template, Response, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')

# Load YOLO models
vehicle_model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy
pedestrian_model = YOLO("yolov8n.pt")

# Classes
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
pedestrian_class = 'person'

# Thresholds
vehicle_threshold = 6
pedestrian_threshold = 8
stop_threads = False

# Video paths
vehicle_video_path = ""
pedestrian_video_path = ""

# Global counts
vehicle_count = 0
pedestrian_count = 0

# Pause control
pause_vehicle = threading.Event()
pause_vehicle.set()  # Vehicle video runs by default

# Function to process vehicle video
def count_vehicles(video_path):
    global vehicle_count, stop_threads
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened() and not stop_threads:
        pause_vehicle.wait()  # Pause vehicle video if event is cleared

        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model(frame, verbose=False)[0]
        count = 0
        for box in results.boxes:
            cls = vehicle_model.names[int(box.cls[0])]
            if cls in vehicle_classes:
                count += 1
        vehicle_count = count
        frame = results.plot()

        # Store the frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame_data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    cap.release()

# Function to process pedestrian video
def count_pedestrians(video_path):
    global pedestrian_count, stop_threads
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened() and not stop_threads:
        ret, frame = cap.read()
        if not ret:
            break

        height = frame.shape[0]
        results = pedestrian_model(frame, verbose=False)[0]
        count = 0
        for box in results.boxes:
            cls = pedestrian_model.names[int(box.cls[0])]
            ymin = int(box.xyxy[0][1])
            if cls == pedestrian_class and ymin > height * 0.65:
                count += 1
        pedestrian_count = count
        frame = results.plot()

        # Store the frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame_data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')  # Make sure this template exists

@app.route('/vehicle_feed')
def vehicle_feed():
    return Response(count_vehicles(vehicle_video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pedestrian_feed')
def pedestrian_feed():
    return Response(count_pedestrians(pedestrian_video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_vehicle', methods=['POST'])
def upload_vehicle():
    global vehicle_video_path
    file = request.files['file']
    if file:
        filename = os.path.join('uploads', 'vehicle_video.mp4')
        file.save(filename)
        vehicle_video_path = filename
        return jsonify(status="success", message="Vehicle video uploaded")
    return jsonify(status="error", message="No file uploaded"), 400

@app.route('/upload_pedestrian', methods=['POST'])
def upload_pedestrian():
    global pedestrian_video_path
    file = request.files['file']
    if file:
        filename = os.path.join('uploads', 'pedestrian_video.mp4')
        file.save(filename)
        pedestrian_video_path = filename
        return jsonify(status="success", message="Pedestrian video uploaded")
    return jsonify(status="error", message="No file uploaded"), 400

@app.route('/start', methods=['POST'])
def start_processing():
    global stop_threads
    stop_threads = False
    threading.Thread(target=count_vehicles, args=(vehicle_video_path,), daemon=True).start()
    threading.Thread(target=count_pedestrians, args=(pedestrian_video_path,), daemon=True).start()
    return jsonify(status="success", message="System started")

@app.route('/status', methods=['GET'])
def get_status():
    # Traffic signal logic
    traffic_flow = "Normal Flow: GREEN for Vehicles | RED for Pedestrians"
    if pedestrian_count >= pedestrian_threshold:
        traffic_flow = "Signal: GREEN for Pedestrians | RED for Vehicles"
    elif vehicle_count >= vehicle_threshold:
        traffic_flow = "Traffic Congestion Detected. Forcing GREEN for Vehicles!"
    elif vehicle_count < 5 and pedestrian_count > 0:
        traffic_flow = "Low vehicle density. Giving GREEN to Pedestrians for 20 seconds."

    return jsonify(vehicle_count=vehicle_count, pedestrian_count=pedestrian_count, traffic_flow=traffic_flow)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
