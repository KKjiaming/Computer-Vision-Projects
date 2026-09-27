import glob
import json
import os

import yaml

from .base_metric import GT, BaseMetric, ImageResult, Pred


class CementMetric(BaseMetric):

    def __init__(self, config_path):
        with open(config_path) as yaml_f:
            config = yaml.load(yaml_f, Loader=yaml.FullLoader)
        self.model_path = config['asphalt_cement_defect']['cement_defect'][
            'model_path']
        self.model_name = os.path.basename(self.model_path).replace(
            '.onnx', '')

        self.gt_defect_names = ['kengdong', 'liefeng', 'mianbanposui', 'bianjiaoboluo', 'banjiaoduanlie', 'xiubu']
        self.pred_defect_names = ['kengdong', 'liefeng', 'mianbanposui', 'bianjiaoboluo', 'banjiaoduanlie', 'xiubu']
        self.ignore_names = ['-1']

        self.ori_liefeng_name = ['liefeng']
        # self.ori_liefeng_name = [
        #     'zongxiangliefeng', 'hengxiangliefeng', 'xiubu', 'junlie',
        #     'kuaizhuangliefeng', 'liefeng','tiefengxiubu', '-1'
        # ]
        self.confidence_threshold = 0.45
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.process_images = []
        self.roi_y_pixel = 2056 - 950
        self.iou_threshold = 0.1

    def init(self, test_folder):
        self.test_folder = test_folder
        test_folder = '/' + test_folder + '/'
        self.pred_path = '/dataset/result/' + self.model_name + test_folder
        self.gt_path = '/dataset/label/巡检测试集/水泥测试集/' + test_folder
        self.image_path = '/dataset/path/to/local_asset' + test_folder

    def match_label(self, shape1, shape2):
        if shape1['label'] == shape2['label']:
            return True
        if shape1['label'] in self.ori_liefeng_name and shape2[
                'label'] in self.ori_liefeng_name:
            return True
        return False

    def run(self, vis_badcase=False, send_message=False, test_folder=None):

        text = ''
        if test_folder is not None:
            self.init(test_folder)
            text = '测试模型:' + self.model_name + '\n'
            text += '测试文件夹:' + test_folder + '\n'
            text += '阈值:' + str(self.confidence_threshold) + '\n'

        gts = {}
        for gt_file in glob.glob(
                os.path.join(self.gt_path, '**', '*.json'), recursive=True):
            gt_data = json.load(open(gt_file))
            image_path = gt_file.replace(self.gt_path,
                                         self.image_path).replace(
                                             '.json', '.jpg')
            gts[os.path.basename(gt_file)] = GT(image_path, gt_data)

        preds = {}
        for pred_file in glob.glob(
                os.path.join(self.pred_path, '**', '*.json'), recursive=True):
            pred_data = json.load(open(pred_file))
            image_path = pred_file.replace(self.pred_path,
                                           self.image_path).replace(
                                               '.json', '.jpg')
            preds[os.path.basename(pred_file)] = Pred(image_path, pred_data)
        results = self.run_metric(gts, preds, self.confidence_threshold,
                                  self.roi_y_pixel, self.iou_threshold)

        fn = 0
        fp = 0
        tp = 0
        for image_name in results.keys():
            fn += results[image_name].num_fns()
            fp += results[image_name].num_fps()
            tp += results[image_name].num_tps()

        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        text += 'tp: ' + str(tp) + ' fp: ' + str(fp) + ' fn: ' + str(fn) \
            + '\n' + 'Precision: ' + \
            str(prec) + ' Recall: ' + str(recall) + '\n'
        print(text)

        arr = self.parseResult(results)
        finalRes = self.countTypes(arr)
        for name in finalRes:
            text = name + '\n'
            fr = finalRes[name]
            prec = fr['tps'] / (fr['tps'] + fr['fps'])
            recall = fr['tps'] / (fr['tps'] + fr['fns'])
            text += 'tp: ' + str(fr['tps']) + ' fp: ' + str(fr['fps']) + ' fn: ' + str(fr['fns']) \
                    + '\n' + 'Precision: ' + str(prec) + ' Recall: ' + str(recall) + '\n'
            print(text)

        if vis_badcase:
            self.visualization(results, self.image_path)
        if send_message:
            self.send_message(text)

    def run_metric(self,
                   gts,
                   preds,
                   confidence_threshold=0.25,
                   roi_y_pixel=1100,
                   iou_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.roi_y_pixel = roi_y_pixel
        self.iou_threshold = iou_threshold
        results = {}
        for image_name in gts.keys():
            gt_data = gts[image_name].data()
            if image_name not in preds.keys():
                for shape in gt_data['shapes']:
                    if self.is_in_roi(shape) and shape[
                            'label'] in self.pred_defect_names:
                        self.fn += 1
                        if image_name not in results.keys():
                            results[image_name] = ImageResult(
                                gts[image_name].path())
                        results[image_name].fns.append(shape)
                continue

            pred_data = preds[image_name].data()
            self.process_images.append(image_name)
            pred_counted = []
            label_counted = []
            # 正常逻辑,满足iou即为tp
            for gt_i, gt_shape in enumerate(gt_data['shapes']):
                if self.is_in_roi(gt_shape) and gt_shape[
                    'label'] in self.gt_defect_names:
                    has_match = False
                    match_preds = []
                    match_indexs = []
                    for pred_i, pred_shape in enumerate(pred_data['shapes']):
                        if float(pred_shape['score']
                                 ) < self.confidence_threshold:
                            continue
                        if pred_shape['label'] not in self.pred_defect_names:
                            continue
                        if self.cal_iou(
                                gt_shape, pred_shape
                        ) >= self.iou_threshold and self.match_label(
                            pred_shape, gt_shape):
                            has_match = True
                            if pred_i not in pred_counted:
                                match_indexs.append(pred_i)
                                match_preds.append(pred_shape)
                    if has_match:
                        if gt_shape['label'] not in self.ignore_names:
                            self.tp += len(match_preds)
                            label_counted.append(gt_i)
                            if image_name not in results.keys():
                                results[image_name] = ImageResult(
                                    gts[image_name].path())
                            for match_pred, match_index in zip(
                                    match_preds, match_indexs):
                                pred_counted.append(match_index)
                                results[image_name].tps.append(match_pred)
                    else:
                        if gt_shape['label'] not in self.ignore_names:
                            self.fn += 1
                            if image_name not in results.keys():
                                results[image_name] = ImageResult(
                                    gts[image_name].path())
                            results[image_name].fns.append(gt_shape)

            # 融合gt里包含gt的情况
            for gt_i, gt_shape in enumerate(gt_data['shapes']):
                if self.is_in_roi(gt_shape) and gt_shape[
                    'label'] in self.gt_defect_names and gt_i not in label_counted and \
                        gt_shape[
                            'label'] not in self.ignore_names:
                    has_match = False
                    for pred_i, pred_shape in enumerate(pred_data['shapes']):
                        if float(pred_shape['score']
                                 ) < self.confidence_threshold:
                            continue
                        if self.cal_iou(gt_shape, pred_shape,
                                        True) >= 0.6 and self.match_label(pred_shape, gt_shape) and pred_shape[
                            'label'] in self.pred_defect_names:
                            has_match = True
                            if pred_i not in pred_counted:
                                pred_counted.append(pred_i)
                    if has_match and gt_shape not in self.ignore_names:
                        self.tp += 1
                        self.fn -= 1
                        label_counted.append(gt_i)
                        if image_name not in results.keys():
                            results[image_name] = ImageResult(
                                gts[image_name].path())
                        results[image_name].tps.append(gt_shape)
                        del results[image_name].fns[
                            results[image_name].fns.index(gt_shape)]

            # 框套框和ignore逻辑
            for pred_i, pred_shape in enumerate(pred_data['shapes']):
                if float(pred_shape['score']) < self.confidence_threshold:
                    continue
                if pred_i not in pred_counted and pred_shape[
                    'label'] in self.pred_defect_names and self.is_in_roi(
                    pred_shape):
                    match_gt = None
                    for gt_i, gt_shape in enumerate(gt_data['shapes']):
                        if self.match_label(pred_shape, gt_shape) and (self.cal_iou(pred_shape, gt_shape) >= self.iou_threshold or self.cal_iou(
                            pred_shape, gt_shape, True) >= 0.6):
                            match_gt = gt_shape
                            break
                    if match_gt is not None:
                        if match_gt['label'] not in self.ignore_names:
                            self.tp += 1
                            if image_name not in results.keys():
                                results[image_name] = ImageResult(
                                    gts[image_name].path())
                            results[image_name].tps.append(pred_shape)
                    else:
                        self.fp += 1
                        if image_name not in results.keys():
                            results[image_name] = ImageResult(
                                gts[image_name].path())
                        results[image_name].fps.append(pred_shape)

        for image_name in preds.keys():
            if image_name not in self.process_images:
                pred_data = preds[image_name].data()
                for shape in pred_data['shapes']:
                    if float(shape['score']) < self.confidence_threshold:
                        continue
                    if self.is_in_roi(shape) and shape[
                        'label'] in self.pred_defect_names:
                        self.fp += 1
                        if image_name not in results.keys():
                            results[image_name] = ImageResult(
                                preds[image_name].path())
                        results[image_name].fps.append(pred_shape)
        return results

    def parseResult(self, results):
        arr = []
        for re in results.keys():
            signle = json.loads('{}')
            signle['imagePath'] = re

            fns = []
            for fn in results[re].fns:
                p_fn = {}
                if 'score' in fn.keys():
                    p_fn['score'] = fn['score']
                p_fn['label'] = fn['label']
                fns.append(p_fn)
            signle['fns'] = fns

            fps = []
            for fp in results[re].fps:
                p_fp = {}
                if 'score' in fp.keys():
                    p_fp['score'] = fp['score']
                p_fp['label'] = fp['label']
                fps.append(p_fp)
            signle['fps'] = fps

            tps = []
            for tp in results[re].tps:
                p_tp = {}
                if 'score' in tp.keys():
                    p_tp['score'] = tp['score']
                p_tp['label'] = tp['label']
                tps.append(p_tp)
            signle['tps'] = tps
            arr.append(signle)

        return arr

    def countTypes(self, j):
        # j = json.loads(res)
        r = json.loads('{}')
        temp = {'xiubu': {'fns': 0, 'fps': 0, 'tps': 0}}
        for itor in j:
            for fn in itor['fns']:
                if fn['label'] not in temp:
                    temp[fn['label']] = {'fns': 0, 'fps': 0, 'tps': 0}
                temp[fn['label']]['fns'] += 1

            for fp in itor['fps']:
                if fp['label'] not in temp:
                    temp[fp['label']] = {'fns': 0, 'fps': 0, 'tps': 0}
                temp[fp['label']]['fps'] += 1

            for tp in itor['tps']:
                if tp['label'] not in temp:
                    temp[tp['label']] = {'fns': 0, 'fps': 0, 'tps': 0}
                temp[tp['label']]['tps'] += 1

        r = json.dumps(temp, indent=4)
        with open('res.json', 'w')as f:
            f.write(r)

        return temp