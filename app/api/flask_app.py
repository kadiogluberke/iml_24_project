from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
import base64
import time
import mediapipe as mp
import os
import signal
import pickle

# Flask app
app = Flask(__name__, static_folder='static')

# Model loading --> for docker
with open('models/random_forest_best/random_forest_best.pkl', 'rb') as file:
    clf = pickle.load(file)

# Model loading --> for local usage
# with open('../models/random_forest_best/random_forest_best.pkl', 'rb') as file:
#     clf = pickle.load(file)

# Integers to characters mapping
int2char = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'd', 27: 's', 28: 'n'
}

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Main route
@app.route('/')
def index():
    with app.app_context():
        images = [
            url_for('static', filename='images/sign_alphabet_1.jpg'),
            url_for('static', filename='images/sign_alphabet_2.png')
        ]
        return render_template('index.html', images=images)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = base64.b64decode(image_data)
    image_np = np.frombuffer(image, dtype=np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    clas = None
    start_time = time.time()
    last_prediction_time = start_time

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    points.append([x, y, z])
        else:
            points = None
            print('Points are None')
        
        if points is not None:
            print('Points are not None')
            print(len(points))
            if len(points) > 21:
                points = points[:21]
                print('Landmarks are more than 21')
            elif len(points) < 21:
                dif = 21 - len(points)
                for i in range(dif):
                    points.append([0, 0, 0])
                print('Landmarks are less than 21')
            
            points_raw = np.array(points)
            
            min_x = np.min(points_raw[:, 0])
            max_x = np.max(points_raw[:, 0])
            min_y = np.min(points_raw[:, 1])
            max_y = np.max(points_raw[:, 1])
            for i in range(len(points_raw)):
                points_raw[i][0] = (points_raw[i][0] - min_x) / (max_x - min_x)
                points_raw[i][1] = (points_raw[i][1] - min_y) / (max_y - min_y)
            
            current_time = time.time()
            # print(f"current_time: {current_time}")
            # print(f"last_prediction_time: {last_prediction_time}")
            if current_time - last_prediction_time >= 0:
                print("Classification started")
                flattened_points = [item for sublist in points_raw for item in sublist]
                X = np.array(flattened_points).reshape(1, -1)
                prediction = clf.predict(X)
                clas = int2char[prediction[0]]
                last_prediction_time = current_time

                print(clas)
    
    return jsonify({'prediction': clas})

# Shutdown route
@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def shutdown_server():
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
