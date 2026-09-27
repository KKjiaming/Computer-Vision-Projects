import cv2
import numpy as np
import onnxruntime


class BaseModule(object):

    def __init__(self):
        self.build_session
        pass

    def build_session(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(
            model_path, providers=providers)

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def input_transform(self, image, mean, std):
        image = cv2.resize(image, None, fx=0.25, fy=0.25)
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= mean
        image /= std
        return image
