# neccessary imports
import cv2
import imutils
import numpy as np
import dlib

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

def defineRectangleCoordinates(bottom_x, bottom_y, top_x, top_y):    
    padding = 20    
    
    w = (top_x + padding) - (bottom_x - padding)
    h_old = (bottom_y + padding) - bottom_x
    h_new = (6.0 * w) / 9.0

    if (h_new >= h_old):
        top_y_new = top_y - (h_new - h_old)
    else:
        top_y_new = top_y + (h_old - h_new)

    rect_point1 = (bottom_x - padding, bottom_y + padding)
    rect_point2 = (top_x + padding, int(top_y_new))

    return (rect_point1, rect_point2)


# main Function
if __name__=="__main__":
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: https://drive.google.com/file/d/1XvAobn_6xeb8Ioa8PBnpCXZm8mgkBTiJ/view?usp=sharing
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 0 means your default web cam
    vid = cv2.VideoCapture(0)

    crop_width = 90
    crop_height = 60
    

    while True:
        _,frame = vid.read()

        # resizing frame
        frame = imutils.resize(frame, width=800)

        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # detecting faces
        face_boundaries = face_detector(frame_gray,0)

        for (enum,face) in enumerate(face_boundaries):

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
            
            crop_left_eye = frame[left_eye_point1[1]-crop_height:left_eye_point1[1], left_eye_point2[0]-crop_width:left_eye_point2[0]]
            crop_right_eye = frame[right_eye_point1[1]-crop_height:right_eye_point1[1], right_eye_point2[0]-crop_width:right_eye_point2[0]]
            
            crop_left_eye_gray = cv2.cvtColor(crop_left_eye, cv2.COLOR_BGR2GRAY)
            crop_right_eye_gray = cv2.cvtColor(crop_right_eye, cv2.COLOR_BGR2GRAY)

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
