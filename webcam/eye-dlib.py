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

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(36, 48):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords

def definePadding(coordinate, percPadding=6):
    return int((coordinate * percPadding)/100)

def defineRectangleCoordinates(bottom_x, bottom_y, top_x, top_y):
    left_eye_point1 = (bottom_x - definePadding(bottom_x), bottom_y + definePadding(bottom_y, 9))
    left_eye_point2 = (top_x + definePadding(top_x), top_y - definePadding(top_y, 9))

    return (left_eye_point1, left_eye_point2)

# main Function
if __name__=="__main__":
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 0 means your default web cam
    vid = cv2.VideoCapture(0)


    while True:
        _,frame = vid.read()

        # resizing frame
        # you can use cv2.resize but I recommend imutils because its easy to use
        frame = imutils.resize(frame, width=800)

        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # detecting faces
        face_boundaries = face_detector(frame_gray,0)

        for (enum,face) in enumerate(face_boundaries):
            # let's first draw a rectangle on the face portion of image
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            # Drawing Rectangle on face part
            cv2.rectangle(frame, (x,y), (x+w, y+h), (120,160,230),2)

            # Now when we have our ROI(face area) let's
            # predict and draw landmarks
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

            #left_eye_point1 = (left_eye_bottom_x - definePadding(left_eye_bottom_x), left_eye_bottom_y + definePadding(left_eye_bottom_y, 9))
            #left_eye_point2 = (left_eye_top_x+definePadding(left_eye_top_x), left_eye_top_y-definePadding(left_eye_top_y, 9))

            crop_img = frame[left_eye_point1[1]-60:left_eye_point1[1], left_eye_point2[0]-90:left_eye_point2[0]]
            crop_img_right = frame[right_eye_point1[1]-60:right_eye_point1[1], right_eye_point2[0]-90:right_eye_point2[0]]
            #cv2.rectangle(frame, left_eye_point1, left_eye_point2, (255, 0, 0), 2)
            #cv2.rectangle(frame, right_eye_point1, right_eye_point2, (255, 0, 0), 2)

            for (a,b) in landmarks:
                # Drawing points on face
                cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)
                #cv2.rectangle(frame, (a, b), (a+w, y+h), (255, 0, 0), 2)

            # Writing face number on image
            cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

        cv2.imshow("frame", frame)
        cv2.imshow("crop", crop_img)
        cv2.imshow("crop_right", crop_img_right)


        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break;
    video_capture.release()
    cv2.destroyAllWindows()
