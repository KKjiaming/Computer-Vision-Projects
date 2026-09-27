import json
import os

import cv2
import numpy as np
import requests
import yaml


class ImageResult(object):

    def __init__(self, image_path):
        self._path = image_path
        self.fns = []
        self.fps = []
        self.tps = []

    def path(self, ):
        return self._path

    def num_fns(self):
        return len(self.fns)

    def num_fps(self):
        return len(self.fps)

    def num_tps(self):
        return len(self.tps)


class GT(object):

    def __init__(self, image_path, json_data):
        self._path = image_path
        self._data = json_data

    def path(self, ):
        return self._path

    def data(self, ):
        return self._data


class Pred(object):

    def __init__(self, image_path, json_data):
        self._path = image_path
        self._data = json_data

    def path(self, ):
        return self._path

    def data(self, ):
        return self._data


class BaseMetric(object):

    def __init__(self, config_path):
        with open(config_path) as yaml_f:
            self.config = yaml.load(yaml_f, Loader=yaml.FullLoader)

    def init(self, test_folder):
        raise NotImplementedError

    def send_message(self, text):
        webhook = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2a8f70ed-732c-4bae-a398-bfa261d98278'
        header = {'Content-Type': 'application/json', 'Charset': 'UTF-8'}
        message = {
            'msgtype': 'text',
            'text': {
                'content': text
            },
            'at': {
                'isAtAll': True
            }
        }
        message_json = json.dumps(message)
        requests.post(url=webhook, data=message_json, headers=header)

    def draw_result(self, shape, image, flag='gt'):
        if flag == 'origin_image':
            x1, y1, x2, y2 = 10, 10, 300, 50
            gt_color = (255, 255, 255)

            image = cv2.line(image, (0, self.roi_y_pixel),
                             (2464, self.roi_y_pixel), (0, 255, 0), 3)
            image = cv2.rectangle(image, (int(x1), int(y1)),
                                  (int(x2), int(y2)), gt_color, 5)
            image = cv2.putText(image, 'gt', (int(
                (x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, gt_color, 1)

            y1 += 60
            y2 += 60
            fp_color = (0, 0, 255)
            image = cv2.rectangle(image, (int(x1), int(y1)),
                                  (int(x2), int(y2)), fp_color, 5)
            image = cv2.putText(image, 'fp', (int(
                (x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, fp_color, 1)

            y1 += 60
            y2 += 60
            tp_color = (0, 255, 0)
            image = cv2.rectangle(image, (int(x1), int(y1)),
                                  (int(x2), int(y2)), tp_color, 5)
            image = cv2.putText(image, 'tp', (int(
                (x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, tp_color, 1)
            y1 += 60
            y2 += 60
            fn_color = (255, 0, 0)
            image = cv2.rectangle(image, (int(x1), int(y1)),
                                  (int(x2), int(y2)), fn_color, 5)
            image = cv2.putText(image, 'fn', (int(
                (x1 + x2) / 2.0), int((y1 + y2) / 2.0)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, fn_color, 1)
            return image

        points = np.array(shape['points'])
        x1, y1, x2, y2 = (min(points[0, 0],
                              points[1, 0]), min(points[0, 1], points[1, 1]),
                          max(points[0, 0],
                              points[1, 0]), max(points[0, 1], points[1, 1]))
        color = (0, 0, 0)
        if flag == 'gt':
            color = (255, 255, 255)  # 白色
        if flag == 'fp':
            color = (0, 0, 255)  #
        if flag == 'tp':
            color = (0, 255, 0)  #
        if flag == 'fn':
            color = (255, 0, 0)  #
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 3)
        if 'score' not in shape.keys():
            image = cv2.putText(image, shape['label'], (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
        else:
            image = cv2.putText(image, shape['label'] + str(shape['score']),
                                (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,
                                1, color, 1)
        return image

    def is_in_roi(self, shape):
        points = np.array(shape['points'])
        x1, y1, x2, y2 = (min(points[0, 0],
                              points[1, 0]), min(points[0, 1], points[1, 1]),
                          max(points[0, 0],
                              points[1, 0]), max(points[0, 1], points[1, 1]))

        _, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cy > self.roi_y_pixel:
            return True
        return False

    def cal_iou(self, shape1, shape2, shape1_self_iou=False):
        # shape1_self_iou 自占比开关
        points1 = np.array(shape1['points'])

        rec1 = [
            min(points1[0, 0], points1[1, 0]),
            min(points1[0, 1], points1[1, 1]),
            max(points1[0, 0], points1[1, 0]),
            max(points1[0, 1], points1[1, 1])
        ]
        rec1[1] = max(self.roi_y_pixel, rec1[1])
        rec1[3] = max(self.roi_y_pixel, rec1[3])

        points2 = np.array(shape2['points'])
        rec2 = [
            min(points2[0, 0], points2[1, 0]),
            min(points2[0, 1], points2[1, 1]),
            max(points2[0, 0], points2[1, 0]),
            max(points2[0, 1], points2[1, 1])
        ]
        rec2[1] = max(self.roi_y_pixel, rec2[1])
        rec2[3] = max(self.roi_y_pixel, rec2[3])
        # computing area of each rectangles
        s_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        s_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        # computing the sum_area
        sum_area = s_rec1 + s_rec2
        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        if shape1_self_iou:
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return (intersect / s_rec1) * 1.0
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / (sum_area - intersect) * 1.0

    def over_dark(self, txt_path):
        if not os.path.exists(txt_path):
            return True
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            data = float(lines[0])
            if data > 50:
                return False
            else:
                return True

    def run_metric(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def visualization(self, results, stuff):
        for image_name in results.keys():
            res = results[image_name]
            image_path = res.path()
            image = cv2.imread(image_path)
            image = self.draw_result(None, image, 'origin_image')
            has_fn = False
            has_fp = False
            for fn in res.fns:
                image = self.draw_result(fn, image, 'fn')
                has_fn = True
            for fp in res.fps:
                image = self.draw_result(fp, image, 'fp')
                has_fp = True
            for tp in res.tps:
                image = self.draw_result(tp, image, 'tp')
            if has_fn:
                vis_path = './debug/' + self.test_folder + '/fn/' + image_path[
                    len(stuff):]
            if has_fp:
                vis_path = './debug/' + self.test_folder + '/fp/' + image_path[
                    len(stuff):]
            if not has_fp and not has_fn:
                vis_path = './debug/' + self.test_folder + '/others/' + image_path[
                    len(stuff):]
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, image)
