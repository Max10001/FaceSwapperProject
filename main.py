import cv2
import numpy as np
import dlib

JAW_POINTS = list(range(0, 17))
NOSE_POINTS = list(range(27, 35))
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

ALIGN_POINTS = (LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS + MOUTH_POINTS + NOSE_POINTS)

OVERLAY_POINTS = (
            LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Path to shape predictor file
DLIB_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_CASCADE = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)

# Our landpoints' predictor and detector objects
predictor = dlib.shape_predictor(DLIB_PATH)
detector = dlib.get_frontal_face_detector()  ##  returns a list of rectangles, each of which corresponding with a face in the image.


# Defining classes for some exception
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


# Detect landpoints' on input image
def get_landmarks(image, use_dlib):
    '''
    Returns a 68x2 element matrix, each row of which corresponding with the
    x, y coordinates of a particular feature point in image.
    '''
    if use_dlib == True:
        points = detector(image, 1)

        if len(points) > 1:
            raise TooManyFaces
        if len(points) == 0:
            raise NoFaces

        return np.matrix([[t.x, t.y] for t in predictor(image, points[0]).parts()])

    else:
        points = face_cascade.detectMultiScale(image, 1.3, 5)
        if len(points) > 1:
            return 'error'
        if len(points) == 0:
            return 'error'
        x, y, w, h = points[0]
        area = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        return np.matrix([[t.x, t.y] for t in predictor(image, area).parts()])