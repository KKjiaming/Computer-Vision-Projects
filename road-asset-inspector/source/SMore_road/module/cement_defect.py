import cv2
import numpy as np
import torch
import torchvision

from .base_module import BaseModule


def inside(box1, box2):
    box1 = box1[0:4]
    box2 = box2[:, 0:4]

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    box2 = box2.T
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    inter = np.clip(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0, 100000) \
        * np.clip(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0, 100000)

    oinds = np.where(inter >= 0.8 * box_area(box1))
    if len(oinds[0]) != 0:
        return oinds, False
    else:
        inds = np.where(inter >= 0.8 * box_area(box2))
        return inds, True


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


class CementDefect(BaseModule):

    def __init__(self, config):
        self.img_size = (1280, 1280)
        self.model_path = config['model_path']
        self.build_session(self.model_path)
        self.defect_names = config['defect_names']

    def run(self, image, confidence_threshold=None):
        ori_shape = image.shape
        image = self.preprocess(image)
        det_shape = image.shape
        output = self.session.run(
            None, {self.session.get_inputs()[0].name: image[None]})[0]
        output = torch.tensor(output).to('cuda')
        out = self.postprocess(
            output, confidence_threshold=confidence_threshold)[0]
        out[:, :4] = self.scale_coords(det_shape[1:], out[:, :4],
                                       ori_shape[:2]).round()
        boxes = []
        scores = []
        labels = []
        for i in range(out.shape[0]):
            o = out[i]
            x1, y1, x2, y2, score, cls = o
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(self.defect_names[int(cls)])
            scores.append(score)
        boxes = np.array(boxes)
        scores = np.array(scores)
        return boxes, scores, labels

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def postprocess(self, preds, confidence_threshold=None):
        if confidence_threshold is None:
            out = self.non_max_suppression(preds)
        else:
            out = self.non_max_suppression(
                preds, conf_thres=confidence_threshold)
        out = [
            self.remove_inside_box(
                out[0],
                classes=[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20
                ])
        ]
        return out

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter
                        )  # iou = inter / (area1 + area2 - inter)

    def remove_inside_box(self, dets, classes=[0, 1, 2, 3, 4]):
        '''
        remove the smaller box inside bigger box from each class
        '''
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        if len(dets) == 0:
            return dets

        out = dets[0]
        for c in classes:
            boxes = dets[dets[:, 5] == c]
            while len(boxes) > 0:
                res, cflag = inside(
                    boxes[0], boxes[1:]
                )  # return inside box index and whether cbox is inside box
                if len(res[0]) != 0:
                    if cflag:
                        boxes = np.delete(boxes, res[0] + 1, axis=0)
                    else:
                        boxes = np.delete(boxes, 0, axis=0)
                else:
                    out = np.vstack((out, boxes[0]))
                    boxes = np.delete(boxes, 0, axis=0)
        return out[1:]

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            multi_confs={}):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        # insert by zhinanzhang 20220317
        multi_conf = len(multi_confs) > 0

        output = [torch.zeros(
            (0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1),
              4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()),
                              1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()),
                              1)  # [conf.view(-1) > conf_thres]
                # insert by zhinanzhang 20220317
                if multi_conf:
                    res = x[0].unsqueeze(0)
                    for c in multi_confs:
                        each_thres = multi_confs[c]
                        each_inds = torch.where(j == c)[0]
                        each_conf, each_box, each_class = conf[each_inds], box[
                            each_inds], j[each_inds]
                        each_x = torch.cat(
                            (each_box, each_conf, each_class.float()),
                            1)[each_conf.view(-1) > each_thres]
                        res = torch.cat((res, each_x), 0)
                    x = res[1:]
                else:
                    x = x[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes,
                                                 device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(
                    descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :
                              4] + c, x[:,
                                        4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n <
                          3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
        return output

    def preprocess(self, image):
        h0, w0 = image.shape[:2]  # orig hw
        r = self.img_size[0] / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                image, (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_LINEAR)
        img = self.letterbox(
            img, self.img_size, auto=False, scaleup=False, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    def letterbox(self,
                  im,
                  new_shape=(640, 640),
                  color=(114, 114, 114),
                  auto=True,
                  scaleFill=False,
                  scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)  # add border
        return im, ratio, (dw, dh)
