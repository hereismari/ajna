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

parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-crop-eyes', type=str, default='shape_predictor_68_face_landmarks.dat',
                    help='download it from: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing')
parser.add_argument('--eye-shape', type=int, nargs="+", default=[90, 60])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NHWC')
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                    default=2, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                    default=1, help='Size of the queue.')
args = parser.parse_args()  


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

def get_inv_landmark_transform(landmarks, frame_gray):
    # Segment eyes
    oh, ow = tuple(args.eye_shape)
    # for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
    for corner1, corner2, is_left in [(42, 45, False)]: # (36, 39, True), 
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            continue
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # Centre image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Rotate to be upright
        roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
        rotate_mat = np.asmatrix(np.eye(3))
        cos = np.cos(-roll)
        sin = np.sin(-roll)
        rotate_mat[0, 0] = cos
        rotate_mat[0, 1] = -sin
        rotate_mat[1, 0] = sin
        rotate_mat[1, 1] = cos
        inv_rotate_mat = rotate_mat.T

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        # Centre image
        centre_mat = np.asmatrix(np.eye(3))
        centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_centre_mat = np.asmatrix(np.eye(3))
        inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                inv_centre_mat)
        eye_image = cv2.warpAffine(frame_gray, transform_mat[:2, :], (ow, oh))
        if is_left:
            eye_image = np.fliplr(eye_image)
        '''
        eyes.append({
            'image': eye_image,
            'inv_landmarks_transform_mat': inv_transform_mat,
            'side': 'left' if is_left else 'right',
        })
        '''
        return inv_transform_mat


def detect_eye_landmarks(image_np, datasource, preprocessor, sess, model):
    preprocessed_image = preprocessor.preprocess_entry(image_np)
    datasource.image = preprocessed_image
    datasource.run_single(sess)
    model.eval(sess)
    input_image, eye_landmarks, eye_heatmaps, eye_radius = model.run_model(sess)
    assert np.all(input_image == preprocessed_image), (input_image, preprocessed_image)
    return eye_landmarks, eye_heatmaps, eye_radius

def setup():
    # Load a Tensorflow model into memory.
    # If needed froze the graph to get better performance.
    data_format = args.data_format
    shape = tuple(args.eye_shape)
    preprocessor = ImgPreprocessor(data_format)
    datasource = ImgDataSource(shape=shape,
                               data_format=data_format)
    # Get model
    model = CNN(datasource.tensors, datasource.x_shape, None,
                data_format=data_format, predict_only=True)

    # Start session
    saver = tf.train.Saver()
    sess = tf.Session()

    # Init variables
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_l)
    # Restore model checkpoint
    saver.restore(sess, args.model_checkpoint)
    
    return datasource, preprocessor, sess, model

def worker(input_q, output_q):
    datasource, preprocessor, sess, model = setup()
    while True:
        x = input_q.get()
        output_q.put(detect_eye_landmarks(x, datasource, preprocessor, sess, model))
    sess.close()


if __name__ == '__main__':
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    shape = (args.eye_shape[1], args.eye_shape[0])

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source).start()
    fps = FPS().start()

    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()
    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: https://drive.google.com/firun_prele/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing
    landmark_predictor = dlib.shape_predictor(args.model_crop_eyes)

    crop_height = 90
    crop_width = 60
    gaze_history = []

    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        
        # resizing frame
        frame = imutils.resize(frame, width=800)

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
        
        aux_height = 30
        aux_width = 20
        crop_left_eye = frame[left_eye_point1[1]-crop_height:left_eye_point1[1]+aux_height, left_eye_point2[0]-crop_width-aux_width:left_eye_point2[0]+aux_width]
        crop_right_eye = frame[right_eye_point1[1]-crop_height:right_eye_point1[1]+aux_height, right_eye_point2[0]-crop_width-aux_width:right_eye_point2[0]+aux_width]
        
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

        eye_landmarks, heatmaps, eye_radius = output_q.get()
        eye_landmarks = eye_landmarks.reshape(18, 2)
        for (a, b) in landmarks.reshape(-1, 2):
            cv2.circle(frame, (a, b), 2, (0, 255, 0), -1)
        
        
        heatmaps_amax = np.amax(heatmaps.reshape(-1, 18), axis=0)
        can_use_eye = np.all(heatmaps_amax > 0.7)
        can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
        can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)
        # eye_index = output['eye_index'][j]
        bgr = frame_rgb
        # bgr = cv2.flip(bgr, flipCode=1)
        eye_image = graph_input
        eye_side = 'right'
        #eye_landmarks = landmarks
        #eye_radius = radius
        if eye_side == 'left':
            eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
            eye_image = np.fliplr(eye_image)

        # Embed eye image and annotate for picture-in-picture
        eye_upscale = 2
        eye_image_raw = cv2.cvtColor(cv2.equalizeHist(eye_image), cv2.COLOR_GRAY2BGR)
        eye_image_raw = cv2.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
        eye_image_annotated = np.copy(eye_image_raw)

        print(eye_landmarks.shape)
        print(eye_landmarks)
        if can_use_eyelid:
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
                                                            .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA,
            )
        if can_use_iris:
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
                                                            .reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
            )
            cv2.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
                color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=1, line_type=cv2.LINE_AA,
            )

            # face_index = int(eye_index / 2)
            face_index = 0
            eh, ew, _ = eye_image_raw.shape
            v0 = face_index * 2 * eh
            v1 = v0 + eh
            v2 = v1 + eh
            u0 = 0 if eye_side == 'left' else ew
            u1 = u0 + ew
            bgr[v0:v1, u0:u1] = eye_image_raw
            bgr[v1:v2, u0:u1] = eye_image_annotated

            face = (face.left(), face.top(), face.right(), face.bottom())

            # Visualize preprocessing results
            frame_landmarks = landmarks
            for landmark in frame_landmarks[:-1]:
                cv2.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                                color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                                markerSize=2, thickness=1, line_type=cv2.LINE_AA)
                cv2.rectangle(
                    bgr, tuple(np.round(face[:2]).astype(np.int32)),
                    tuple(np.round(face[2:]).astype(np.int32)),
                    color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,
                )

            eye_landmarks = np.concatenate([eye_landmarks, [[eye_landmarks[-1, 0] + eye_radius, eye_landmarks[-1, 1]]]])
            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                'constant', constant_values=1.0))
            print(eye_landmarks.shape)
            eye_landmarks = (eye_landmarks *
                                get_inv_landmark_transform(landmarks, frame_gray).T)[:, :2]
            print(eye_landmarks.shape)
            eye_landmarks = np.asarray(eye_landmarks)
            eyelid_landmarks = eye_landmarks[0:8, :]
            iris_landmarks = eye_landmarks[8:16, :]
            iris_centre = eye_landmarks[16, :]
            eyeball_centre = eye_landmarks[17, :]
            eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                            eye_landmarks[17, :])

            print(eyeball_centre, type(eyeball_centre[0]))
            print(iris_centre)
            if can_use_eye:
                # Visualize landmarks
                #cv2.drawMarker(  # Eyeball centre
                #    bgr, tuple(np.round(eyeball_centre.astype(np.float32)).astype(np.int32)),
                #    color=(0, 255, 0),
                    # markerType=cv2.MARKER_CROSS, markerSize=4,
                    #thickness=1, line_type=cv2.LINE_AA,
                #)
                # cv2.circle(  # Eyeball outline
                #     bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                #     int(np.round(eyeball_radius)), color=(0, 255, 0),
                #     thickness=1, lineType=cv2.LINE_AA,
                # )

                # Draw "gaze"
                # from models.hg_gaze import estimate_gaze_from_landmarks
                # current_gaze = estimate_gaze_from_landmarks(
                #     iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
                i_x0, i_y0 = iris_centre
                e_x0, e_y0 = eyeball_centre
                theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                        -1.0, 1.0))
                current_gaze = np.array([theta, phi])
                gaze_history.append(current_gaze)
                gaze_history_max_len = 10
                if len(gaze_history) > gaze_history_max_len:
                    gaze_history = gaze_history[-gaze_history_max_len:]
                print('Gaze')
                print(current_gaze)
                util.draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
                               length=120.0, thickness=1)       
        
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        cv2.imshow("frame", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()