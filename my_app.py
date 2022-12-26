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
socketio = SocketIO(app, cors_allowed_origins="*", message_queue='amqp://')
# socketio = SocketIO(app, cors_allowed_origins="*")

# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

celery = make_celery(app)

global rec, rec_frame, p, detector, predictor, display_alert, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR, EAR_data, initial_EAR, initial_mean, initial_sd, original_time, Num_Frame
display_alert = 0
drowsiness_value = -1
drowsiness_val_submitted = 0
rec = 0


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
    return random.randint(200, 5000)

@celery.task(name="task.message")
def save_frames(data_image, Num_Frame, display_alert, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR_json, initial_EAR, initial_mean, initial_sd, original_time, EAR_data):
    # global rec, display_alert, p, detector, predictor, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR, EAR_data, initial_EAR, initial_mean, initial_sd, original_time

    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # converting RGB to BGR, as opencv standards
    image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    print("Frame# ", Num_Frame)

    # Load All Models
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    panda_EAR = pd.read_json(panda_EAR_json)

    now_time = time.perf_counter()  # used to measure time
    elapsed_time_secs = now_time - original_time

    if image.any():
        # print("====================== inside image.any() ====================")
        print(image[0][0])
        Num_Frame = Num_Frame + 1  # For frame rate measurement

        # getting height and width
        h, w, _ = image.shape

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(gray, 0)

        # Ignore frames which has no detected face
        if len(rects) == 0:
            print("====================== inside rects ====================")
            EAR_data.append((Num_Frame, None, elapsed_time_secs, 'Undefined'))
            # continue

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            print("====================== inside enumerate ====================")
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

            if Num_Frame <= 50:
                initial_EAR.append(average_EAR)
            if Num_Frame == 50:
                # mean and sd
                initial_mean = statistics.mean(initial_EAR)
                initial_sd = statistics.stdev(initial_EAR)
                # Blink_Status records if there was a blink or not

            Blink_Status = 'Undefined'

            if Num_Frame > 50:
                if average_EAR < initial_mean - 2 * initial_sd:
                    x1 = int(w / 2) - 20
                    Blink_Status = 'Blink'
                else:
                    Blink_Status = 'Not Blink'

            EAR_data.append((Num_Frame, average_EAR, elapsed_time_secs, Blink_Status))
            print(EAR_data)

            # panda_EAR = pd.concat([panda_EAR, pd.DataFrame.from_records([{
            #     'Frame Number': Num_Frame, 'Average EAR': average_EAR,
            #     'Elapsed Time': elapsed_time_secs,
            #     'Response': Blink_Status, 'Actual Output': drowsiness_value}
            # ])], ignore_index=True)

            # # Remove the drowsiness_value after adding once
            # if drowsiness_value:
            #     drowsiness_value = -1

            # if drowsiness_val_submitted == 1:
            #     display_alert = 0
            #     drowsiness_val_submitted = 0

            # if Num_Frame > 200:
            #     # randomly sending 5 alerts
            #     print("Num_Frame: ", Num_Frame)
            #     if Num_Frame in random_alert_frames:
            #         print("================= Frame Matched ===================")
            #         print(Num_Frame)
            #         if drowsiness_val_submitted == 0:
            #             print("drowsiness_val_submitted: ", drowsiness_val_submitted)
            #             display_alert = 1
            return EAR_data


# @celery.task
def detect_landmarks(image):
    global rec, display_alert, p, detector, predictor, drowsiness_value, drowsiness_val_submitted

    random_alert_frames = [random_number(), random_number(), random_number(), random_number(), random_number()]
    print("============random_alert_frames=============")
    print(random_alert_frames)

    panda_EAR = pd.DataFrame(columns=['Frame Number', 'Average EAR', 'Elapsed Time', 'Response', 'Actual Output'])
    EAR_data = []
    initial_EAR = []

    initial_mean = math.inf
    initial_sd = math.inf
    original_time = time.perf_counter()  # used to measure time

    # To measure frame rate
    Num_Frame = 0

    while True:
        # print("================== rec ======================")
        # print(rec)

        if not rec:
            now = datetime.datetime.now()
            panda_EAR.to_csv('panda_EAR_{}.csv'.format(str(now).replace(":", '')))
            break

        # measuring time
        now_time = time.perf_counter()  # used to measure time
        elapsed_time_secs = now_time - original_time

        # Getting out image by webcam
        # global rec_frame
        # image = rec_frame

        if image.any():
            Num_Frame = Num_Frame + 1  # For frame rate measurement

            # getting height and width
            h, w, _ = image.shape

            # Converting the image to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get faces into webcam's image
            rects = detector(gray, 0)

            # Ignore frames which has no detected face
            if len(rects) == 0:
                EAR_data.append((Num_Frame, None, elapsed_time_secs, 'Undefined'))
                # continue

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

                if Num_Frame <= 50:
                    initial_EAR.append(average_EAR)
                    # cv2.putText(image, 'Training, Try not to blink', (int(w / 2) - 180, h - 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.2, (255, 0, 0), 2, cv2.LINE_AA)
                if Num_Frame == 50:
                    # mean and sd
                    initial_mean = statistics.mean(initial_EAR)
                    initial_sd = statistics.stdev(initial_EAR)
                    # Blink_Status records if there was a blink or not

                Blink_Status = 'Undefined'

                if Num_Frame > 50:
                    if average_EAR < initial_mean - 2 * initial_sd:
                        x1 = int(w / 2) - 20
                        # cv2.putText(image, 'Blink', (x1, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        #             cv2.LINE_AA)
                        Blink_Status = 'Blink'
                    else:
                        Blink_Status = 'Not Blink'

                EAR_data.append((Num_Frame, average_EAR, elapsed_time_secs, Blink_Status))

                # panda_EAR = panda_EAR.append(
                #     {'Frame Number': Num_Frame, 'Average EAR': average_EAR, 'Elapsed Time': elapsed_time_secs,
                #      'Response': Blink_Status, 'Actual Output': drowsiness_value}, ignore_index=True)

                panda_EAR = pd.concat([panda_EAR, pd.DataFrame.from_records([{
                    'Frame Number': Num_Frame, 'Average EAR': average_EAR,
                    'Elapsed Time': elapsed_time_secs,
                    'Response': Blink_Status, 'Actual Output': drowsiness_value}
                ])], ignore_index=True)

                # Remove the drowsiness_value after adding once
                if drowsiness_value:
                    drowsiness_value = -1

                # cv2.putText(image, 'Frame= ' + str(Num_Frame), (7, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255),
                #             1,
                #             cv2.LINE_AA)
                # cv2.putText(image, 'EAR= ' + str(average_EAR), (w - 125, 28), cv2.FONT_HERSHEY_SIMPLEX, .7,
                #             (0, 255, 125),
                #             1, cv2.LINE_AA)

                # (x, y, w, h) = face_utils.rect_to_bb(rect)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #
                # for (x, y) in shape:
                #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

                # if not (len(rects) == 0 or len(rects) > 1):
                # panda_EAR.to_csv('panda_EAR.csv')
                # now = datetime.datetime.now()
                # panda_EAR.to_csv('panda_EAR_{}.csv'.format(str(now).replace(":", '')))

                # Draw plot
                # rolling = drawPlots.drawPlots()
                # print("After Rolling: ", rolling)

                if drowsiness_val_submitted == 1:
                    display_alert = 0
                    drowsiness_val_submitted = 0

                if Num_Frame > 200:
                    # randomly sending 5 alerts
                    print("Num_Frame: ", Num_Frame)
                    if Num_Frame in random_alert_frames:
                        print("================= Frame Matched ===================")
                        print(Num_Frame)
                        if drowsiness_val_submitted == 0:
                            print("drowsiness_val_submitted: ", drowsiness_val_submitted)
                            display_alert = 1


@app.route('/', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def index():
    global p, detector, predictor

    # Load All Models
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    return render_template('index.html')


# @socketio.on('requests')
@app.route('/requests', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def requests():
    global rec
    print("======================= Before Recording Value =========================================")
    print(rec)
    rec = not rec
    print("======================= Before Recording Value =========================================")
    print(rec)

    global random_alert_frames, panda_EAR, EAR_data, initial_EAR, initial_mean, initial_sd, original_time, Num_Frame

    if rec:
        random_alert_frames = [random_number(), random_number(), random_number(), random_number(), random_number()]
        print("============random_alert_frames=============")
        print(random_alert_frames)

        panda_EAR = pd.DataFrame(columns=['Frame Number', 'Average EAR', 'Elapsed Time', 'Response', 'Actual Output'])
        print("================ panda_EAR initialized ====================")
        EAR_data = []
        initial_EAR = []

        initial_mean = math.inf
        initial_sd = math.inf
        original_time = time.perf_counter()  # used to measure time

        # To measure frame rate
        Num_Frame = 0
    else:
        now = datetime.datetime.now()
        panda_EAR.to_csv('panda_EAR_{}.csv'.format(str(now).replace(":", '')))

    return "success"


@app.route('/getDynamicPort', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def getDynamicPort():
    global rec

    if rec:
        detect_landmarks()

    return "success"


@app.route('/fetch_drowsiness_alert')
@cross_origin(supports_credentials=True)
def fetch_drowsiness_alert():
    global display_alert
    print("display_alert: ", display_alert)
    return str(display_alert)


@app.route('/fetch_drowsiness_alert_values')
@cross_origin(supports_credentials=True)
def fetch_drowsiness_alert_values():
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
    global rec, display_alert, p, detector, predictor, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR, EAR_data, initial_EAR, initial_mean, initial_sd, original_time, Num_Frame

    print("Getting Frames")

    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # converting RGB to BGR, as opencv standards
    image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # print('=============== frame ================')
    # print(image[0][0])

    if rec:
        print("=================== REC is TRUE ===========================")
        # save_frames.delay(image)
        # Convert from type DataFrame to JSON
        # print("================ panda_EAR before ====================")
        # print(panda_EAR)
        panda_EAR_json = panda_EAR.to_json()
        # print("================ panda_EAR after ====================")
        # print(panda_EAR)
        result = save_frames.delay(data_image, Num_Frame, display_alert, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR_json, initial_EAR, initial_mean, initial_sd, original_time, EAR_data)
        print("======================= Before result.get() ==================================")
        print(result.get())
        EAR_data_returned = result.get()

        # panda_EAR = pd.concat([panda_EAR, pd.DataFrame.from_records([{
        #     'Frame Number': Num_Frame, 'Average EAR': average_EAR,
        #     'Elapsed Time': elapsed_time_secs,
        #     'Response': Blink_Status, 'Actual Output': drowsiness_value}
        # ])], ignore_index=True)

        # update Num_Frame
        Num_Frame = EAR_data_returned[0][0] + 1

        # Remove the drowsiness_value after adding once
        # if drowsiness_value:
        #     drowsiness_value = -1

        # if drowsiness_val_submitted == 1:
        #     display_alert = 0
        #     drowsiness_val_submitted = 0

        # if Num_Frame > 200:
        #     # randomly sending 5 alerts
        #     print("Num_Frame: ", Num_Frame)
        #     if Num_Frame in random_alert_frames:
        #         print("================= Frame Matched ===================")
        #         print(Num_Frame)
        #         if drowsiness_val_submitted == 0:
        #             print("drowsiness_val_submitted: ", drowsiness_val_submitted)
        #             display_alert = 1
        print("======================= After result.get() ===================================")

    return "success"


@socketio.on('video_stream')
@cross_origin(supports_credentials=True)
def video_stream(data_image):
    global rec, display_alert, p, detector, predictor, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR, EAR_data, initial_EAR, initial_mean, initial_sd, original_time, Num_Frame

    print("Getting Frames")

    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # converting RGB to BGR, as opencv standards
    image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # print('=============== frame ================')
    # print(image[0][0])

    if rec:
        print("=================== REC is TRUE ===========================")
        # save_frames.delay(image)
        # Convert from type DataFrame to JSON
        # print("================ panda_EAR before ====================")
        # print(panda_EAR)
        panda_EAR_json = panda_EAR.to_json()
        # print("================ panda_EAR after ====================")
        # print(panda_EAR)
        result = save_frames.delay(data_image, Num_Frame, display_alert, drowsiness_value, drowsiness_val_submitted, random_alert_frames, panda_EAR_json, initial_EAR, initial_mean, initial_sd, original_time, EAR_data)
        print("======================= Before result.get() ==================================")
        print(result.get())
        EAR_data_returned = result.get()

        Num_Frame = EAR_data_returned[0][0] + 1

        print("======================= After result.get() ===================================")

    return "success"


@socketio.on('connection')
@cross_origin(supports_credentials=True)
def connect():
    print("======================== Socket connected ========================")

    return "success"


if __name__ == '__main__':
    # socketio.run(app, host='127.0.0.1')
    socketio.run(app, debug=True)
