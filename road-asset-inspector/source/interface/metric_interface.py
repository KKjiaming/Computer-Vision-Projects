import glob
import json
import os
import typing as t
from abc import ABCMeta

from SMore_road.metric.asphalt_metric import AsphaltMetric
from SMore_road.metric.asset_metric import AssetMetric
from SMore_road.metric.base_metric import GT, ImageResult, Pred
from SMore_road.metric.cement_metric import CementMetric

JSONType = t.Union[str, int, float, bool, None, t.Dict[str, t.Any],
                   t.List[t.Any], t.Any]
DataType = t.Dict[str, JSONType]
ResultType = t.Dict[str, ImageResult]


class Request(object):
    """Request.
    Attributes:
        gts (JSONType): gts.
        preds (JSONType): preds.
        confidence_threshold (float): 置信度阈值
        roi_y_pixel (float): 八米线距离最上的像素长度
        iou_threshold (float): iou阈值
    """

    def __init__(self, gts: DataType, preds: DataType,
                 confidence_threshold: float, roi_y_pixel: float,
                 iou_threshold: float):
        self._gts = gts
        self._preds = preds
        self.confidence_threshold = confidence_threshold
        self.roi_y_pixel = roi_y_pixel
        self.iou_threshold = iou_threshold

    @property
    def gts(self) -> DataType:
        return self._gts

    @property
    def preds(self) -> DataType:
        return self._preds


class Response(object):
    """Response.
    Attributes:
        results (ResultType): defect code 0-35.
    """

    def __init__(self, results: ResultType):
        self._results: ResultType = results

    @property
    def results(self) -> ResultType:
        return self._results


class SMoreRoadMetric(metaclass=ABCMeta):
    """Interface class of SMoreRoad."""

    def Init(self):
        """Init Config."""
        self.cement_metric = CementMetric('./config/road.yaml')
        self.asphalt_metric = AsphaltMetric('./config/road.yaml')
        self.asset_metric = AssetMetric('./config/road.yaml')

    def AsphaltMetric(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        gts = {}
        for image_name in req.gts.keys():
            gts[image_name] = GT(image_name, req.gts[image_name])
        preds = {}
        for image_name in req.preds.keys():
            preds[image_name] = Pred(image_name, req.preds[image_name])
        results = self.asphalt_metric.run_metric(gts, preds,
                                                 req.confidence_threshold,
                                                 req.roi_y_pixel,
                                                 req.iou_threshold)
        return Response(results)

    def CementMetric(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        gts = {}
        for image_name in req.gts.keys():
            gts[image_name] = GT(image_name, req.gts[image_name])
        preds = {}
        for image_name in req.preds.keys():
            preds[image_name] = Pred(image_name, req.preds[image_name])
        results = self.cement_metric.run_metric(gts, preds,
                                                req.confidence_threshold,
                                                req.roi_y_pixel,
                                                req.iou_threshold)
        return Response(results)

    def AssetMetric(self, req: Request) -> Response:
        """Do Inference.
        Args:
            req (Request): inference request.
        Returns:
            Response: list of inference response.
        """
        gts = {}
        for image_name in req.gts.keys():
            gts[image_name] = GT(image_name, req.gts[image_name])
        preds = {}
        for image_name in req.preds.keys():
            preds[image_name] = Pred(image_name, req.preds[image_name])
        results = self.asset_metric.run_metric(gts, preds,
                                               req.confidence_threshold,
                                               req.roi_y_pixel,
                                               req.iou_threshold)
        return Response(results)


if __name__ == '__main__':
    smrm = SMoreRoadMetric()
    smrm.Init()
    # 水泥测试
    test_folder = '水泥-泰州11.15'
    test_folder = '/' + test_folder + '/'
    model_name = 'shuini-xinghua1125-check'
    pred_path = '/dataset/result/' + model_name + test_folder
    gt_path = '/dataset/label/巡检测试集/水泥测试集/' + test_folder
    image_path = '/dataset/path/to/local_asset' + test_folder

    gts = {}
    for gt_file in glob.glob(
            os.path.join(gt_path, '**', '*.json'), recursive=True):
        gt_data = json.load(open(gt_file))
        gts[os.path.basename(gt_file)] = gt_data

    preds = {}
    for pred_file in glob.glob(
            os.path.join(pred_path, '**', '*.json'), recursive=True):
        pred_data = json.load(open(pred_file))
        preds[os.path.basename(pred_file)] = pred_data
    req = Request(gts, preds, 0.25, 2056 - 950, 0.1)
    res = smrm.CementMetric(req)
    results = res.results

    fn = 0
    fp = 0
    tp = 0
    for image_name in results.keys():
        fn += results[image_name].num_fns()
        fp += results[image_name].num_fps()
        tp += results[image_name].num_tps()
    text = ''
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    text += 'tp: ' + str(tp) + ' fp: ' + str(fp) + ' fn: ' + str(fn) \
        + '\n' + 'Precision: ' + \
        str(prec) + ' Recall: ' + str(recall) + '\n'
    print(text)

    # 沥青测试

    test_folder = '11.30广韶高速'
    test_folder = '/' + test_folder + '/'
    model_name = 'asphalt_defect_v0.4.0'
    pred_path = '/dataset/result/' + model_name + test_folder
    gt_path = '/dataset/label/巡检测试集/沥青测试集/' + test_folder
    image_path = '/dataset/path/to/local_asset' + test_folder

    gts = {}
    for gt_file in glob.glob(
            os.path.join(gt_path, '**', '*.json'), recursive=True):
        gt_data = json.load(open(gt_file))
        gts[os.path.basename(gt_file)] = gt_data

    preds = {}
    for pred_file in glob.glob(
            os.path.join(pred_path, '**', '*.json'), recursive=True):
        pred_data = json.load(open(pred_file))
        preds[os.path.basename(pred_file)] = pred_data
    req = Request(gts, preds, 0.1, 2056 - 1150, 0.1)
    res = smrm.AsphaltMetric(req)
    results = res.results

    fn = 0
    fp = 0
    tp = 0
    for image_name in results.keys():
        fn += results[image_name].num_fns()
        fp += results[image_name].num_fps()
        tp += results[image_name].num_tps()
    text = ''
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    text += 'tp: ' + str(tp) + ' fp: ' + str(fp) + ' fn: ' + str(fn) \
        + '\n' + 'Precision: ' + \
        str(prec) + ' Recall: ' + str(recall) + '\n'
    print(text)

    # 资产管理测试
    test_folder = 'demo'
    test_folder = '/' + test_folder + '/'
    model_name = 'asset_detect_v0.5.0'
    pred_path = '/dataset/result/' + model_name + test_folder
    gt_path = '/dataset/label/巡检测试集/交通资产测试集/' + test_folder
    image_path = '/dataset/path/to/local_asset' + test_folder

    gts = {}
    for gt_file in glob.glob(
            os.path.join(gt_path, '**', '*.json'), recursive=True):
        gt_data = json.load(open(gt_file))
        gts[os.path.basename(gt_file)] = gt_data

    preds = {}
    for pred_file in glob.glob(
            os.path.join(pred_path, '**', '*.json'), recursive=True):
        pred_data = json.load(open(pred_file))
        preds[os.path.basename(pred_file)] = pred_data

    req = Request(gts, preds, 0.5, 2056 - 2056, 0.4)
    res = smrm.AssetMetric(req)
    results = res.results

    fn = 0
    fp = 0
    tp = 0
    for image_name in results.keys():
        fn += results[image_name].num_fns()
        fp += results[image_name].num_fps()
        tp += results[image_name].num_tps()
    text = ''
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    text += 'tp: ' + str(tp) + ' fp: ' + str(fp) + ' fn: ' + str(fn) \
        + '\n' + 'Precision: ' + \
        str(prec) + ' Recall: ' + str(recall) + '\n'
    print(text)
