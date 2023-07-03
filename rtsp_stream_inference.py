import cv2
import os
from threading import Thread
import time
import timeit
import base64
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from json import dumps
import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(prog="Program human detection menggunakan SSD MobileNetV2 / YOLOv4-tiny pada Dept. Teknologi Informasi ITS", usage="Human Detection")
parser.add_argument("-m", "--model", default="mobilenet", help="model selection. use mobilenet or yolo")
parser.add_argument("-s", "--stream", default=2, help="stream number. 1=main, 2=sub")
parser.add_argument("-c", "--cctv", default=11, help="1: Lift Gerbang Barat, 2: Selasar Gerbang Barat, 11: Lab KCKS Belakang, 3: Selasar Lab KCKS")
args = parser.parse_args()

API_ENDPOINT = 'http://127.0.0.1:5000/api/footage'
STREAM_NUMBER = args.stream
CCTV_NUMBER = args.cctv
PATH_TO_SAVED_MODEL="../inference_graph_v2_40k/saved_model"
PATH_TO_YOLOV4_WEIGHT = 'D:/Kuliah/Bangkit ML/TA/technical/yolo-coco/data/yolov4-tiny-custom_last.weights'
PATH_TO_YOLOV4_CFG = 'D:/Kuliah/Bangkit ML/TA/technical/yolo-coco/yolov4-tiny-custom.cfg'

url = f"rtsp://KCKS:majuteru5@10.15.40.48:554/Streaming/Channels/{CCTV_NUMBER}0{STREAM_NUMBER}"

class VideoScreenshot(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Take screenshot every x seconds
        self.screenshot_interval = 0.35

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    # # not needed sementara, nanti buat ditampilin di web
    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.inf_frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def mobilenetv2_inference(self):
            self.frame_count = 0
            self.total_numeric_fps = 0

            # Save obtained frame periodically
            def save_frame_thread():
                time.sleep(5)
                while True:
                    try:
                        self.frame_count += 1
                        
                        # Initialize frame rate calculation
                        frame_rate_calc = 1
                        freq = cv2.getTickFrequency()

                        # Start timer (for calculating frame rate)
                        t1 = cv2.getTickCount()

                        self.inf_frame = self.frame.copy()

                        # inferensi
                        frame_rgb = cv2.cvtColor(self.inf_frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (320, 320))
                        image_np = np.array(frame_resized)

                        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                        input_tensor = tf.convert_to_tensor(image_np)
                        # The model expects a batch of images, so add an axis with `tf.newaxis`.
                        input_tensor = input_tensor[tf.newaxis, ...]

                        detections = detect_fn(input_tensor)

                        # All outputs are batches tensors.
                        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                        # We're only interested in the first num_detections.
                        num_detections = int(detections.pop('num_detections'))
                        detections = {key: value[0, :num_detections].numpy()
                                    for key, value in detections.items()}
                        detections['num_detections'] = num_detections

                        # detection_classes should be ints.
                        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                        boxes = detections['detection_boxes']
                        scores = detections['detection_scores']

                        current_count=0
                        for i in range(len(scores)):
                            if ((scores[i] > 0.4) and (scores[i] <= 1.0)):

                                # Get bounding box coordinates and draw box
                                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                                ymin = int(max(1,(boxes[i][0] * self.frame_height)))
                                xmin = int(max(1,(boxes[i][1] * self.frame_width)))
                                ymax = int(min(self.frame_height,(boxes[i][2] * self.frame_height)))
                                xmax = int(min(self.frame_width,(boxes[i][3] * self.frame_width)))
                                
                                cv2.rectangle(self.inf_frame, (xmin,ymin), (xmax,ymax),
                                                color=(0, 255, 0), thickness=1)

                                # Draw label
                                label = 'person: %d%%' % (int(scores[i]*100)) # Example: 'person: 72%'
                                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                                
                                cv2.rectangle(self.inf_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                                cv2.putText(self.inf_frame, 
                                            label, 
                                            (xmin, label_ymin-7), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5,
                                            color=(0, 0, 0), 
                                            thickness=1)
                                current_count+=1

                
                        # catat end timer
                        t2 = cv2.getTickCount()
                        time1 = (t2-t1)/freq
                        frame_rate_calc= 1/time1
                        self.total_numeric_fps += frame_rate_calc

                        # model name
                        cv2.putText(self.inf_frame,
                                    'SSD MobileNetV2',
                                    (15, 375), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5,
                                    (0,255,55), 
                                    1, 
                                    cv2.LINE_AA)
                            
                        # Draw framerate in corner of frame
                        cv2.putText(self.inf_frame,
                                    'FPS: {0:.2f}'.format(frame_rate_calc),
                                    (15, 400), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (0,255,55), 
                                    1, 
                                    cv2.LINE_AA)
                        
                        #avg. FPS
                        cv2.putText(self.inf_frame,
                                    'avg. FPS: {0:.2f}'.format(self.total_numeric_fps/self.frame_count),
                                    (15, 425), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5,
                                    (0,255,55), 
                                    1, 
                                    cv2.LINE_AA)

                        # Num detections
                        cv2.putText (self.inf_frame,
                                        'Total Detection Count : ' + str(current_count),
                                        (15, 450),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0,255,55),
                                        1,
                                        cv2.LINE_AA)
                        

                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
                        ret, frame = cv2.imencode('.jpeg', self.inf_frame, encode_param)

                        # Send to server
                        processed_image = base64.b64encode(frame)
                        values = {
                            "cctv_number"   : CCTV_NUMBER,
                            "num_detections" : current_count,
                            "image" : processed_image.decode('utf-8')
                        }
                        header = {
                            "Content-Type": "application/json"
                        }
                        response = requests.post(
                            url = API_ENDPOINT, 
                            json = values, 
                            headers = header
                        )

                        print(f"avg fps: {round(self.total_numeric_fps/self.frame_count, 2)}")
                        time.sleep(self.screenshot_interval)
                    except KeyboardInterrupt:
                        pass
            self.inf_thread = Thread(target=save_frame_thread, args=())
            self.inf_thread.daemon = True
            self.inf_thread.start()

    def yolov4tiny_inference(self):
        self.frame_count = 0
        self.total_numeric_fps = 0

        # Save obtained frame periodically
        def save_frame_thread():
            time.sleep(5)
            while True:
                self.frame_count += 1

                # Inference time
                #start = timeit.default_timer()
                
                # Initialize frame rate calculation
                frame_rate_calc = 1
                freq = cv2.getTickFrequency()

                # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()

                self.inf_frame = self.frame.copy()

                # lakukan inferensi
                classIds, scores, boxes = model.detect(self.inf_frame, confThreshold=0.4, nmsThreshold=0.4)

                current_count=0
                for (classId, score, box) in zip(classIds, scores, boxes):
                    cv2.rectangle(self.inf_frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                color=(0, 0, 255), thickness=1)
                    
                    text = 'person: %.2f' % (score)
                    labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size 
                    label_ymin = max(box[1], labelSize[1] + 10) # Make sure not to draw label too close to top of window

                    cv2.rectangle(self.inf_frame, (box[0], label_ymin-labelSize[1]-10), 
                                (box[0]+labelSize[0], 
                                label_ymin+baseLine-10), 
                                (255, 255, 255), 
                                cv2.FILLED) # Draw white box to put label text in     
                    
                    cv2.putText(self.inf_frame, 
                                text, 
                                (box[0], box[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                color=(0, 0, 0), 
                                thickness=1)
                    current_count += 1
                
                #stop = timeit.default_timer()

                # catat end timer
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc= 1/time1
                self.total_numeric_fps += frame_rate_calc

                # model name
                cv2.putText(self.inf_frame,
                            'YOLOv4-tiny',
                            (15, 10+375), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5,
                            (0,255,55), 
                            1, 
                            cv2.LINE_AA)
                    
                # Draw framerate in corner of frame
                cv2.putText(self.inf_frame,
                            'FPS: {0:.2f}'.format(frame_rate_calc),
                            (15, 10+400), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0,255,55), 
                            1, 
                            cv2.LINE_AA)
                
                #avg. FPS
                cv2.putText(self.inf_frame,
                            'avg. FPS: {0:.2f}'.format(self.total_numeric_fps/self.frame_count),
                            (15, 10+425), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5,
                            (0,255,55), 
                            1, 
                            cv2.LINE_AA)

                # Num detections
                cv2.putText (self.inf_frame,
                                'Total Detection Count : ' + str(current_count),
                                (15, 10+450),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,55),
                                1,
                                cv2.LINE_AA)
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
                ret, frame = cv2.imencode('.jpeg', self.inf_frame, encode_param)

                # Send to server
                processed_image = base64.b64encode(frame)
                values = {
                    "cctv_number"   : CCTV_NUMBER,
                    "num_detections" : current_count,
                    "image" : processed_image.decode('utf-8')
                }
                header = {
                    "Content-Type": "application/json"
                }
                response = requests.post(
                    url = API_ENDPOINT, 
                    json = values, 
                    headers = header
                )

                print(f"avg fps: {round(self.total_numeric_fps/self.frame_count, 2)}")
                time.sleep(self.screenshot_interval)

        self.inf_thread = Thread(target=save_frame_thread, args=())
        self.inf_thread.daemon = True
        self.inf_thread.start()

if __name__ == '__main__':
    if args.model == "mobilenet":
        print('Loading model SSD MobileNetV2...', end='')
        # Load saved model and build the detection function
        detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
        print('Done!')
    else:
        print('Loading model YOLOv4-tiny...', end='')
        net = cv2.dnn.readNetFromDarknet(PATH_TO_YOLOV4_CFG, PATH_TO_YOLOV4_WEIGHT)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(320, 320), swapRB=True)
        print('Done!')

    # Buat objek baru untuk melakukan inferensi
    video_stream_widget = VideoScreenshot(url)

    # melakukan inferensi
    if args.model == "mobilenet":
        video_stream_widget.mobilenetv2_inference()
    else:
        video_stream_widget.yolov4tiny_inference()
        
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass