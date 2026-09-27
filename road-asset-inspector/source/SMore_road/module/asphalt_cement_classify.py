import cv2
import numpy as np

from .base_module import BaseModule


class AsphaltCementClassify(BaseModule):

    def __init__(self, config):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.scale = np.float32(1.0 / 255.0)
        self.input_WH = [864, 224]
        self.crop_height = 1416
        self.model_path = config['model_path']
        self.defect_names = config['defect_names']
        self.build_session(self.model_path)

    def run(self, image):
        model_input = self.preprocess(image)
        model_output = self.session.run(
            None, {self.session.get_inputs()[0].name: model_input})[0][0]
        output = self.postprocess(model_output)
        return output

    def preprocess(self, image):
        """
        image: input image.shape is (H, W, 3) and BGR format
        process_steps:
        1:BGR2RGB
        2:crop image
        3:resize image
        4:normalize image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_image = image[1416:]
        resize_image = cv2.resize(
            crop_image, self.input_WH, interpolation=cv2.INTER_LINEAR)
        norm_image = (resize_image.astype('float32') * self.scale -
                      self.mean) / self.std
        norm_image = norm_image[np.newaxis].transpose(0, 3, 1, 2)
        return norm_image

    def postprocess(self, output):
        cls = output.argmax(axis=0)
        return self.defect_names[int(cls)]


if __name__ == '__main__':
    config = {'model_path': '/dataset/model/road_classify.onnx'}
    model = AsphaltCementClassify(config)
    image = cv2.imread('test-image/shuini.jpg')
    print(model.run(image))
    image = cv2.imread('test-image/liqing.jpg')
    print(model.run(image))
