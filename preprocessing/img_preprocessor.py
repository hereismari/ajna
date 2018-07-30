import cv2 as cv
import numpy as np
from preprocessing.preprocessor import Preprocessor as BasePreprocessor

class ImgPreprocessor(BasePreprocessor):
    def __init__(self, data_format='NHWC'):
        super().__init__(data_format=data_format)

    def preprocess_entry(self, entry):
        eye = cv.equalizeHist(entry)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if self.data_format == 'NHWC' else 0)
        return eye