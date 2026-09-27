import typing as t
from abc import ABCMeta

import cv2
import numpy as np
from SMore_road.engine import RoadEngine

JSONType = t.Union[str, int, float, bool, None, t.Dict[str, t.Any],
                   t.List[t.Any]]

NameType = t.List[str]
ConveredType = t.List[bool]


class Request(object):
    """Request.
    Attributes:
        image (np.ndarray): image np.ndarry.
    """

    def __init__(self, image: np.ndarray):
        self._image = image

    @property
    def image(self) -> np.ndarray:
        return self._image


class Response(object):
    """Response.
    Attributes:
        boxes (np.ndarray): 2-dimensional numpy array, each row represents the bounding box [x1,y1,x2,y2] of a target
        scores (np.ndarray): confidences of each target
        label_names (NameType): labels of each target
        is_covered (ConveredType): represent if it is covered of each asset target
        ocr_name (NameType): ocrresult of each asset target
        retrival_name (NameType): retrival classfication of each asset target
    """

    def __init__(self,
                 boxes: np.ndarray,
                 scores: np.ndarray,
                 label_names: NameType,
                 is_covered: ConveredType = None,
                 ocr_name: NameType = None,
                 retrival_name: NameType = None):
        self._boxes: np.ndarray = boxes
        self._scores: np.ndarray = scores
        self._label_names: NameType = label_names
        self._is_covered: bool = is_covered
        self._ocr_name: NameType = ocr_name
        self._retrival_name: NameType = retrival_name

    @property
    def boxes(self) -> np.ndarray:
        return self._boxes

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @property
    def label_names(self) -> NameType:
        return self._label_names

    @property
    def is_covered(self) -> ConveredType:
        return self._is_covered

    @property
    def ocr_name(self) -> NameType:
        return self._ocr_name

    @property
    def retrival_name(self) -> NameType:
        return self._retrival_name


class SMoreRoad(metaclass=ABCMeta):
    """Interface class of SMoreRoad."""

    def Init(self):
        """Init Config."""
        self.road_engine = RoadEngine('./config/road.yaml')

    def AsphaltDefectRun(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        boxes, scores, labels = self.road_engine.asphalt_defect_run(req.image)
        res = Response(boxes, scores, labels)
        return res

    def CementDefectRun(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        boxes, scores, labels = self.road_engine.cement_defect_run(req.image)
        res = Response(boxes, scores, labels)
        return res

    def AutoDefectRun(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        boxes, scores, labels = self.road_engine.defect_auto_run(req.image)
        res = Response(boxes, scores, labels)
        return res

    def AssetDefectRun(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        boxes, scores, labels, is_covered, ocr_name, retrival_name = self.road_engine.asset_run(
            req.image)
        res = Response(boxes, scores, labels, is_covered, ocr_name,
                       retrival_name)
        return res


if __name__ == '__main__':
    # image = cv2.imread(
    #     '/dataset/path/to/local_asset.jpg'
    # )
    image = cv2.imread('/dataset/path/to/local_asset.jpg')
    smr = SMoreRoad()
    smr.Init()
    req = Request(image)

    # smr.AutoDefectRun(req)

    # smr.CementDefectRun(req)

    # smr.AsphaltDefectRun(req)

    smr.AssetDefectRun(req)
