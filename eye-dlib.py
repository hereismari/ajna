# neccessary imports
import cv2
import imutils
import util.util as util
import numpy as np
import dlib

import sys
sys.path.append('../..')

import os
import time
import multiprocessing
import tensorflow as tf

from webcam.fps import FPS
from webcam.webcam import WebcamVideoStream
from multiprocessing import Queue, Pool

NUM_CLASSES = 90

from data_sources.img_data_source import ImgDataSource
from preprocessing.img_preprocessor import ImgPreprocessor
from models.cnn import CNN
from learning.trainer import Trainer

import argparse
parser = argparse.ArgumentParser(description='Webcam')

parser.add_argument('--landmarks', type=str, default='model', choices=['model', 'dlib68'])
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--eye-shape', type=int, nargs="+", default=[90, 60])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NHWC')


# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the eyes landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(36, 48):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords


def defineRectangleCoordinates(bottom_x, bottom_y, top_x, top_y, width=6., height=9.):    
    padding = 20    
    
    w = (top_x + padding) - (bottom_x - padding)
    h_old = (bottom_y + padding) - bottom_x
    h_new = (width * w) / height

    if (h_new >= h_old):
        top_y_new = top_y - (h_new - h_old)
    else:
        top_y_new = top_y + (h_old - h_new)

    rect_point1 = (bottom_x - padding, bottom_y + padding)
    rect_point2 = (top_x + padding, int(top_y_new))

    return (rect_point1, rect_point2)


def get_evaluator(args):
    datasource = ImgDataSource(shape=tuple(args.eye_shape),
                               data_format=args.data_format)
    # Get model
    learning_schedule=[
        {
            'loss_terms_to_optimize': {
                'heatmaps_mse': ['hourglass'],
                'radius_mse': ['radius'],
            },
            'learning_rate': 1e-3,
        }
    ]

    model = CNN(datasource.tensors, datasource.x_shape, learning_schedule, data_format=args.data_format, predict_only=True)
    evaluator = Trainer(model, model_checkpoint=args.model_checkpoint)
    return datasource, evaluator

def predict(evaluator, image, datasource, preprocessor):
    preprocessed_image = preprocessor.preprocess_entry(image)
    datasource.image = preprocessed_image
    output, _ = evaluator.run_predict(datasource)    
    import ipdb; ipdb.set_trace()
    util.plot_predictions2(output, preprocessed_image.reshape(90, 60))


def detect_eye_landmarks(image_np, datasource, preprocessor, sess, model):
    image_np = preprocessor.preprocess_entry(image_np)
    datasource.image = image_np
    datasource.eval.run_single(sess)
    print(image_np)
    eye_landmarks = model.run_model(sess)
    print(eye_landmarks)
    return eye_landmarks


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:input_q
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    '''
    data_format = 'NHWC'
    shape = (60, 90)
    preprocessor = ImgPreprocessor(data_format)
    datasource = ImgDataSource(shape=shape,
                               data_format=data_format)
    # Get model
    learning_schedule=[
        {
            'loss_terms_to_optimize': {
                'heatmaps_mse': ['hourglass'],
                'radius_mse': ['radius'],
            },
            'learning_rate': 1e-3,
        }
    ]

    model = CNN(datasource.tensors, datasource.x_shape, learning_schedule,
                data_format=data_format, predict_only=True)

    saver = tf.train.Saver()
    sess = tf.Session()

    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_l)

    saver.restore(sess, 'checkpoints/best_cnn.ckpt')
    model.eval(sess)

    fps = FPS().start()
    
    while True:
        fps.update()
        x = input_q.get()
        output_q.put(detect_eye_landmarks(x, datasource, preprocessor, sess, model))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()


    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)


    data_format = 'NHWC'
    shape = (60, 90)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()
    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    crop_height = 90
    crop_width = 60

    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting faces
        face_boundaries = face_detector(frame_gray, 0)
        if len(face_boundaries) < 1:
            continue
        face = face_boundaries[0]
        # for face in face_boundaries:
        # Let's predict and draw landmarks
        landmarks = landmark_predictor(frame_gray, face)

        # converting co-ordinates to NumPy array
        landmarks = land2coords(landmarks)

        left_eye_bottom_x = landmarks[36][0]
        left_eye_bottom_y = landmarks[41][1]

        left_eye_top_x = landmarks[39][0]
        left_eye_top_y = landmarks[38][1]

        left_eye_point1, left_eye_point2 = defineRectangleCoordinates(left_eye_bottom_x, left_eye_bottom_y, left_eye_top_x, left_eye_top_y)

        right_eye_bottom_x = landmarks[42][0]
        right_eye_bottom_y = landmarks[45][1]

        right_eye_top_x = landmarks[45][0]
        right_eye_top_y = landmarks[44][1]

        right_eye_point1, right_eye_point2 = defineRectangleCoordinates(right_eye_bottom_x, right_eye_bottom_y, right_eye_top_x, right_eye_top_y)
        
        aux_height = 0
        aux_width = 0
        crop_left_eye = frame[left_eye_point1[1]-crop_height:left_eye_point1[1]+aux_height, left_eye_point2[0]-crop_width-aux_width:left_eye_point2[0]+aux_width]
        crop_right_eye = frame[right_eye_point1[1]-crop_height+20:right_eye_point1[1]+aux_height+20, right_eye_point2[0]-crop_width-aux_width:right_eye_point2[0]+aux_width]
        
        crop_left_eye = cv2.resize(crop_left_eye, shape)
        crop_right_eye = cv2.resize(crop_right_eye, shape)

        crop_left_eye_gray = cv2.cvtColor(crop_left_eye, cv2.COLOR_BGR2GRAY)
        crop_right_eye_gray = cv2.cvtColor(crop_right_eye, cv2.COLOR_BGR2GRAY)

        graph_input = crop_right_eye_gray
        input_q.put(graph_input)

        # detect_eye_landmarks(crop_right_eye_gray, datasource, preprocessor, sess, model)
        nchannels = 1
        crop_left_eye_gray = np.resize(crop_left_eye_gray, (crop_height, crop_width, nchannels))
        crop_right_eye_gray = np.resize(crop_right_eye_gray, (crop_height, crop_width, nchannels))

        frame[0:crop_left_eye.shape[0], 0:crop_left_eye.shape[1]] = crop_left_eye_gray
        frame[0:crop_right_eye.shape[0], crop_width+2:crop_width+2+crop_right_eye.shape[1]] = crop_right_eye_gray

        for (a,b) in landmarks:
            # Drawing points on face
            cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)

        t = time.time()

        x = output_q.get()
        print(x)

        util.plot_predictions2(x, graph_input.reshape(90, 60))
        for (a, b) in x[0]:
            cv2.circle(frame, (a, b), 2, (0, 255, 0), -1)
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()


# main Function
'''
if __name__=="__main__":
    args = parser.parse_args()
    shape = (args.eye_shape[1], args.eye_shape[0])

    datasource, evaluator = get_evaluator(args)
    preprocessor = ImgPreprocessor(args.data_format)

    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # 0 means your default web cam
    vid = cv2.VideoCapture(0)

    crop_height = args.eye_shape[0]
    crop_width = args.eye_shape[1]
    
    while True:
        _,frame = vid.read()

        # resizing frame
        frame = imutils.resize(frame, width=800)

        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # detecting faces
        face_boundaries = face_detector(frame_gray, 0)
        for (enum, face) in enumerate(face_boundaries):
            # Let's predict and draw landmarks
            landmarks = landmark_predictor(frame_gray, face)

            # converting co-ordinates to NumPy array
            landmarks = land2coords(landmarks)

            left_eye_bottom_x = landmarks[36][0]
            left_eye_bottom_y = landmarks[41][1]

            left_eye_top_x = landmarks[39][0]
            left_eye_top_y = landmarks[38][1]

            left_eye_point1, left_eye_point2 = defineRectangleCoordinates(left_eye_bottom_x, left_eye_bottom_y, left_eye_top_x, left_eye_top_y)

            right_eye_bottom_x = landmarks[42][0]
            right_eye_bottom_y = landmarks[45][1]

            right_eye_top_x = landmarks[45][0]
            right_eye_top_y = landmarks[44][1]

            right_eye_point1, right_eye_point2 = defineRectangleCoordinates(right_eye_bottom_x, right_eye_bottom_y, right_eye_top_x, right_eye_top_y)
            
            aux_height = 30
            aux_width = 20
            crop_left_eye = frame[left_eye_point1[1]-crop_height:left_eye_point1[1]+aux_height, left_eye_point2[0]-crop_width-aux_width:left_eye_point2[0]+aux_width]
            crop_right_eye = frame[right_eye_point1[1]-crop_height:right_eye_point1[1]+aux_height, right_eye_point2[0]-crop_width-aux_width:right_eye_point2[0]+width]
            
            crop_left_eye = cv2.resize(crop_left_eye, shape)
            crop_right_eye = cv2.resize(crop_right_eye, shape)

            crop_left_eye_gray = cv2.cvtColor(crop_left_eye, cv2.COLOR_BGR2GRAY)
            crop_right_eye_gray = cv2.cvtColor(crop_right_eye, cv2.COLOR_BGR2GRAY)

            predict(evaluator, crop_right_eye_gray, datasource, preprocessor)

            nchannels = 1
            crop_left_eye_gray = np.resize(crop_left_eye_gray, (crop_height, crop_width, nchannels))
            crop_right_eye_gray = np.resize(crop_right_eye_gray, (crop_height, crop_width, nchannels))

            frame[0:crop_left_eye.shape[0], 0:crop_left_eye.shape[1]] = crop_left_eye_gray
            frame[0:crop_right_eye.shape[0], crop_width+2:crop_width+2+crop_right_eye.shape[1]] = crop_right_eye_gray

            for (a,b) in landmarks:
                # Drawing points on face
                cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)


        cv2.imshow("frame", frame)

        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break;
    video_capture.release()
    cv2.destroyAllWindows()
'''