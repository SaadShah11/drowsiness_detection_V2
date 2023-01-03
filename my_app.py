# import eventlet
# eventlet.monkey_patch()

from flask_celery import make_celery
from celery import Celery
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

import cv2
import numpy as np
import base64
import io
from io import StringIO
from PIL import Image
import dlib
import math
import face_utils
import pandas as pd
import random
import statistics
from flask_cors import CORS, cross_origin
import datetime, time
import User
# import urllib.parse
# import os
# from threading import Thread
# import imutils

app = Flask(__name__)
CORS(app, support_credentials=True)
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['CELERY_BROKER_URL'] = 'amqp://localhost//'
app.config['CELERY_RESULT_BACKEND'] = 'rpc://'
# Get latest port from redis and add in socketio
# global port = 
# socketio = SocketIO(app, cors_allowed_origins="*", message_queue='redis://')
socketio = SocketIO(app, cors_allowed_origins="*", message_queue='amqp://', async_mode='threading')
# socketio = SocketIO(app, cors_allowed_origins="*")

# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

celery = make_celery(app)

global rec, rec_frame, p, detector, predictor, display_alert, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR, EAR_data, initial_EAR, initial_mean, initial_sd, original_time, Num_Frame
display_alert = 0
drowsiness_value = -1
drowsiness_val_submitted = 0
rec = 0

global users
users = {}

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dis

def eyeAspectRatio(Eye):
    # lets calculate individual euclidean distances
    vertical1 = euclidean_distance(Eye[1], Eye[5])
    vertical2 = euclidean_distance(Eye[2], Eye[4])
    horizontal = euclidean_distance(Eye[0], Eye[3])
    EAR = (vertical1 + vertical2) / (2 * horizontal)

    return EAR

def random_number():
    return random.randint(200, 2000)

@celery.task(name="task.message")
def save_frames(data_image, Num_Frame, original_time, EAR_data, initial_mean, initial_sd, currentSocketID):
    global detector, predictor

    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # converting RGB to BGR, as opencv standards
    image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    print(f"Frame#  {Num_Frame}")

    # Load All Models
    # p = "shape_predictor_68_face_landmarks.dat"
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(p)

    now_time = time.perf_counter()  # used to measure time
    elapsed_time_secs = now_time - original_time

    if image.any():
        # print("============ image.any() =============")
        # getting height and width
        h, w, _ = image.shape

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(gray, 0)

        print(len(rects))

        # Ignore frames which has no detected face
        if len(rects) == 0:
            EAR_data.append((Num_Frame, None, elapsed_time_secs, 'Undefined'))

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # we now store the shapes from 36--41 (inclusive) for left eye
            leftEye = [shape[36], shape[37], shape[38], shape[39], shape[40], shape[41]]
            # right eye shapes from 41--47
            rightEye = [shape[42], shape[43], shape[44], shape[45], shape[46], shape[47]]
            left_EAR = eyeAspectRatio(leftEye)
            right_EAR = eyeAspectRatio(rightEye)
            average_EAR = round((left_EAR + right_EAR) / 2, 2)

            Blink_Status = 'Undefined'

            if Num_Frame > 60:
                if average_EAR < initial_mean - 2 * initial_sd:
                    x1 = int(w / 2) - 20
                    Blink_Status = 'Blink'
                else:
                    Blink_Status = 'Not Blink'

            EAR_data.append((Num_Frame, average_EAR, elapsed_time_secs, Blink_Status, currentSocketID))
            print(EAR_data)

            return EAR_data
    else:
        return None


@app.route('/', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def index():
    return render_template('index.html')


# @socketio.on('requests')
@app.route('/requests', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def requests():
    global users

    currentSocketID = request.args.get('socketID')

    # Get current socketID
    currentUser = users[currentSocketID]
    print('================== currentSocketID =================')
    print(currentSocketID)

    currentUser.rec = not currentUser.rec
    print(currentUser.rec)

    if currentUser.rec:
        random_alert_frames = [random_number(), random_number(), random_number(), random_number(), random_number(), random_number(), random_number(), random_number(), random_number(), random_number()]
        print("============random_alert_frames=============")
        print(random_alert_frames)

        panda_EAR = pd.DataFrame(columns=['Frame Number', 'Average EAR', 'Elapsed Time', 'Response', 'Actual Output'])
        print("================ panda_EAR initialized ====================")
        EAR_data = []
        initial_EAR = []

        initial_mean = math.inf
        initial_sd = math.inf
        original_time = time.perf_counter()  # used to measure time
        print(original_time)
        # To measure frame rate
        Num_Frame = 0

        # Updates these values in the user object in users dictionary
        currentUser.random_alert_frames = random_alert_frames
        # currentUser.rec = rec
        currentUser.display_alert = 0
        currentUser.drowsiness_value = 1
        currentUser.drowsiness_val_submitted = 0
        currentUser.initial_EAR = []
        currentUser.EAR_data = []
        currentUser.initial_mean = math.inf
        currentUser.initial_sd = math.inf
        currentUser.panda_EAR = panda_EAR
        currentUser.original_time = original_time
        currentUser.Num_Frame = 0

        print('============== Currnet User ==============')
        print(currentUser.rec)
    else:
        print("============ Rec Stopped =============")
        # print(panda_EAR)

        now = datetime.datetime.now()
        currentUser.panda_EAR.to_csv('panda_EAR_{}.csv'.format(str(now).replace(":", '')))

        # Delete the user object from users dictionary
        # Add new user object for current socketID again

        # When user with socketID is deleted we get errors in image endpoint 
        # del users[currentSocketID]

        # print('=============== Current user deleted =================')
        # print(users)

    return "success"


@app.route('/fetch_drowsiness_alert')
@cross_origin(supports_credentials=True)
def fetch_drowsiness_alert():
    # Perform this route using sockets

    global display_alert
    print("display_alert: ", display_alert)
    return str(display_alert)


@app.route('/fetch_drowsiness_alert_values')
@cross_origin(supports_credentials=True)
def fetch_drowsiness_alert_values():
    # Perform this route using sockets

    global drowsiness_val_submitted, drowsiness_value
    drowsiness_value = int(request.args.get('drowsiness_value'))
    print("=========================== drowsiness_value =========================")
    print(drowsiness_value)

    if drowsiness_value == 1:
        drowsiness_value = "Extremely Alert"
    elif drowsiness_value == 2:
        drowsiness_value = "Very Alert"
    elif drowsiness_value == 3:
        drowsiness_value = "Alert"
    elif drowsiness_value == 4:
        drowsiness_value = "Fairly Alert"
    elif drowsiness_value == 5:
        drowsiness_value = "Neither Alert Nor Sleepy"
    elif drowsiness_value == 6:
        drowsiness_value = "Some Signs of sleepiness"
    elif drowsiness_value == 7:
        drowsiness_value = "Sleepy, but no effort to keep alert"
    elif drowsiness_value == 8:
        drowsiness_value = "Sleepy, some effort to keep alert"
    elif drowsiness_value == 9:
        drowsiness_value = "Very Sleepy, great effort to keep alert, fighting sleep"
    else:
        drowsiness_value = -1

    print(drowsiness_value)

    drowsiness_val_submitted = 1
    return str(drowsiness_val_submitted)


@socketio.on('image')
@cross_origin(supports_credentials=True)
def image(data_image):
    global users
    # print("Getting Frames")

    # Get current socketID
    currentSocketID = str(request.sid)
    currentUser = users[currentSocketID]
    # print('================== currentSocketID =================')
    # print(currentSocketID)

    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # converting RGB to BGR, as opencv standards
    image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    if currentUser.rec:
        # print("=================== REC is TRUE ===========================")

        currentUser.Num_Frame += 1
        result = save_frames.delay(data_image, currentUser.Num_Frame, currentUser.original_time, currentUser.EAR_data, currentUser.initial_mean, currentUser.initial_sd, currentSocketID)

        EAR_data_returned = result.get()
        # print("====================== EAR_data_returned ====================")
        print(EAR_data_returned)

        if EAR_data_returned:
            if currentUser.Num_Frame <= 50:
                # print("====================== Num_Frame <= 50 ====================")
                currentUser.initial_EAR.append(EAR_data_returned[0][1])

            if currentUser.Num_Frame == 50:
                # print("====================== Num_Frame == 50 ====================")
                # mean and sd
                currentUser.initial_mean = statistics.mean(currentUser.initial_EAR)
                currentUser.initial_sd = statistics.stdev(currentUser.initial_EAR)

            # Remove the drowsiness_value after adding once
            if currentUser.drowsiness_value:
                currentUser.drowsiness_value = -1

            if currentUser.drowsiness_val_submitted == 1:
                currentUser.display_alert = 0
                currentUser.drowsiness_val_submitted = 0

            if currentUser.Num_Frame > 200:
                # randomly sending 5 alerts
                # print("Num_Frame: ", currentUser.Num_Frame)
                if currentUser.Num_Frame in currentUser.random_alert_frames:
                    print("================= Frame Matched ===================")
                    print(currentUser.Num_Frame)
                    if currentUser.drowsiness_val_submitted == 0:
                        print("drowsiness_val_submitted: ", currentUser.drowsiness_val_submitted)
                        currentUser.display_alert = 1

            currentUser.panda_EAR = pd.concat([currentUser.panda_EAR, pd.DataFrame.from_records([{
                'Frame Number': EAR_data_returned[0][0], 'Average EAR': EAR_data_returned[0][1],
                'Elapsed Time': EAR_data_returned[0][2],
                'Response': EAR_data_returned[0][3], 'Actual Output': drowsiness_value}
            ])], ignore_index=True)

    return "success"

@socketio.on('connect')
@cross_origin(supports_credentials=True)
def connect():
    global users
    currentSocketID = str(request.sid)
    print("================= SOCKET ID ==================")
    print(currentSocketID)

    if not users.__contains__(currentSocketID):
        print('=========== New User Added =============')
        users[currentSocketID] = User.User()

    print('============== USERS ===========')
    print(users)

    return "Success"


if __name__ == '__main__':
    # socketio.run(app, host='127.0.0.1')
    socketio.run(app, debug=True)
