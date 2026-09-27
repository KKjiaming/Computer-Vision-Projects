import glob
import json
import os

from .base_metric import GT, BaseMetric, ImageResult, Pred


class AssetMetric(BaseMetric):

    def __init__(self, config_path,confidence_threshold = 0.1, roi_y_pixel = 0, iou_threshold = 0.4, area_threshold =3000):
        super().__init__(config_path)

        self.model_path = self.config['asset']['asset_detect']['detect_model_path']
        self.model_name = os.path.basename(self.model_path).replace(
            '.onnx', '')
        # 清洗后
        # self.gt_defect_names = [
        #     'gonglizhuang', 'jiaotongbiaozhi',
        #     'longmenjia', 'dianziqingbaoban', 'yuanbiaozhi',
        #     'jiaotongxinhaodeng',  'jinggai', 'sanjiaobiaozhi'
        # ]
        self.gt_defect_names = ['lichengzhuang', 'biaozhibiaopai', 'longmenjia', \
            'dianziqingbaoban', 'jiaotongxinhaodeng',  'jinggai', 'sanjiaobiaozhi','covered']
        
        self.confs = {
            'longmenjia': 0.4,
            'dianziqingbaoban': 0.3,
            'jiaotongxinhaodeng': 0.5,
            'jinggai': 0.7,#0.7
            'jiaotongbiaozhi': 0.3,
            'sanjiaobiaozhi': 0.5,
            'yuanbiaozhi': 0.4,
            'gonglizhuang': 0.5,
        }
        self.confidence_threshold = confidence_threshold
        
        # #清洗前
        # self.gt_defect_names = [
        #     'longmenjia', 'dianziqingbaoban', 'jiaotongxinhaodeng', \
        #     'jinggai', 'jiaotongbiaozhi', 'sanjiaobiaozhi', \
        #     'yuanbiaozhi','gonglizhuang'
        #     ]
        
        self.traffic_signs = [
            'yuanbiaozhi', 'sanjiaobiaozhi', 'jiaotongbiaozhi'
        ]
        self.roi_y_pixel = roi_y_pixel
        self.iou_threshold = iou_threshold
        self.area_threshold = area_threshold
        
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.process_images = []


    def init(self, test_folder):
        self.test_folder = test_folder
        test_folder = '/' + test_folder + '/'
        # /dataset/result/asset_detect_v0.6.0/巡检测试集/交通资产测试集/detection_val/json8/image
        # /dataset/result/detection_8_yolov5_s_best/巡检测试集/交通资产测试集/detection_val/json8/image
        # /dataset/result/detection_7_yolov5_m_best/巡检测试集/交通资产测试集/detection_val/json8/image
        # /dataset/result/raod-facility-1280-s6-med-covered/巡检测试集/交通资产测试集/detection_val/json8/image
        self.pred_path ="/dataset/result/raod-facility-1280-s6-med-covered/巡检测试集/交通资产测试集/detection_val/json8/image"
        self.gt_path = '/dataset/label/巡检测试集/交通资产测试集/' + test_folder
        self.image_path = '/dataset/path/to/local_asset' + "detection_val/image/"

    def match_label(self, shape1, shape2):
        #pred ,gt
        if shape1['label'] == shape2['label']:
            return True
        # if shape1['label'] == 'lichengzhuang' and shape2['label'] =='gonglizhuang':
        #     return True
        # if shape1['label'] in 
        
        
        return False
    
    def remove_small_siangs(self, img_data):
        # 遍历所有的图片和json数据
        for image_name in img_data.keys():
            json_data = img_data[image_name].data()   
            # 对于一张图片来说，对每一个小于阈值交通标志进行过滤
            for shape in json_data['shapes']:
                h = shape['points'][1][1] - shape['points'][0][1]
                w = shape['points'][1][0] - shape['points'][0][0]
                area = h * w
                if shape['label'] =="biaozhibiaopai" and area < self.area_threshold:
                    # breakpoint()
                    json_data['shapes'].remove(shape)
            
            if json_data['shapes'] == 0:
                # breakpoint()
                del img_data[image_name]
                
        return img_data

    def run(self, vis_badcase=True, send_message=False, test_folder=None):

        text = ''
        if test_folder is not None:
            self.init(test_folder)
            text = '测试模型:' + self.model_name + '\n'
            text += '测试文件夹:' + test_folder + '\n'
            text += '阈值:' + str(self.confidence_threshold) + '\n'
        # 解析gt和pred中的json
        # gt字典 basename : GT class
        gts = {}
        for gt_file in glob.glob(
                os.path.join(self.gt_path, '**', '*.json'), recursive=True):
            gt_data = json.load(open(gt_file))
            image_path = gt_file.replace(self.gt_path,
                                         self.image_path).replace(
                                             '.json', '.jpg')
            gts[os.path.basename(gt_file)] = GT(image_path, gt_data)

        # pred 字典 basename : Pred class
        preds = {}
        for pred_file in glob.glob(
                os.path.join(self.pred_path, '**', '*.json'), recursive=True):
            pred_data = json.load(open(pred_file))
            image_path = pred_file.replace(self.pred_path,
                                           self.image_path).replace(
                                               '.json', '.jpg')
            preds[os.path.basename(pred_file)] = Pred(image_path, pred_data)
            
        # breakpoint()
        # 过滤所有小于阈值的交通标志
        preds = self.remove_small_siangs(preds)
        gts  = self.remove_small_siangs(gts)
        
        # 计算得到所有的p/r
        results = self.run_metric(gts, preds)
        # 输出所有的结果
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
        
        if vis_badcase:
            self.visualization(results, self.image_path)
        if send_message:
            self.send_message(text)
            
        # 对每一个类别的结果解析
        arr = self.parseResult(results)
        finalRes = self.countTypes(arr)
        
        # breakpoint()
        rs_cls = {'fns': 0, 'fps': 0, 'tps': 0}
        
        for name in finalRes:
            text = name + '\n'
            fr = finalRes[name]
            if name in self.gt_defect_names:
                rs_cls['fns'] += fr['fns']
                rs_cls['fps'] += fr['fps']
                rs_cls['tps'] += fr['tps']
                
            prec = round(fr['tps'] / (fr['tps'] + fr['fps']),4)
            recall = round(fr['tps'] / (fr['tps'] + fr['fns']),4)
            text += 'tp: ' + str(fr['tps']) + ' fp: ' + str(
                fr['fps']) + ' fn: ' + str(fr['fns']) \
                    + '\n' + 'Precision: ' + str(prec) + ' Recall: ' + str(
                recall) + '\n' + f"{format(prec, '.3f')}/{format(recall, '.3f')}" + '\n'
            print(text)
            
        # lftext = 'liefengclass' + '\n'
        # lfprec = liefeng['tps'] / (liefeng['tps'] + liefeng['fps'])
        # lfrecall = liefeng['tps'] / (liefeng['tps'] + liefeng['fns'])
        # lftext += 'tp: ' + str(liefeng['tps']) + ' fp: ' + str(
        #     liefeng['fps']) + ' fn: ' + str(liefeng['fns']) \
        #           + '\n' + 'Precision: ' + str(lfprec) + ' Recall: ' + str(
        #     lfrecall) + '\n' + f"{format(lfprec, '.3f')}/{format(lfrecall, '.3f')}" + '\n'
        # print(lftext)

    def run_metric(self,gts,preds):

        # 字典 图片的名字：一张图的存在的结果
        results = {}
        
        # gt里面所有的图片名字
        for image_name in gts.keys():
            gt_data = gts[image_name].data()                
            # 情况一： gt里面存在图片，但是pred里面不存在这个图片，说明存在漏检fn
            if image_name not in preds.keys():
                # 遍历gt中的每一个label
                for gt_shape in gt_data['shapes']:
                    self.fn += 1
                    # 如果字典不存在，就初始化一条ImageResult
                    # 把这条imageResult存起来
                    if image_name not in results.keys():
                        results[image_name] = ImageResult(
                            gts[image_name].path())
                    results[image_name].fns.append(gt_shape)
                continue
            
            # 情况二：gt里面存在的图片，pred也存在这个图片
            pred_data = preds[image_name].data()
            self.process_images.append(image_name)
            
            # 保存pred每一个label的index
            pred_counted = []
            # 保存gt的每个label的index
            label_counted = []

            # 正常逻辑,满足iou即为tp
            # 遍历gt的每一个label
            for gt_i, gt_shape in enumerate(gt_data['shapes']):
                has_match = False
                # 保存index对应的label数据
                match_preds = []

                for pred_i, pred_shape in enumerate(pred_data['shapes']):
                    if float(pred_shape['score']) < self.confidence_threshold:
                    # if float(pred_shape['score']) < self.confs[pred_shape['label']]:
                        continue
                    if self.cal_iou( gt_shape, pred_shape) >= self.iou_threshold and self.match_label(pred_shape, gt_shape):
                        has_match = True
                        # 存一下这个shape
                        if pred_i not in pred_counted:
                            pred_counted.append(pred_i)
                            match_preds.append(pred_shape)
                #存在匹配上的框
                if has_match:
                    self.tp += len(match_preds)
                    label_counted.append(gt_i)

                    # 初始化一个image
                    if image_name not in results.keys():
                        results[image_name] = ImageResult(
                            gts[image_name].path())

                    for match_pred in match_preds:
                        pred_counted.append(pred_i)

                        results[image_name].tps.append(match_pred)
                else:
                    self.fn += 1
                    if image_name not in results.keys():
                        results[image_name] = ImageResult(
                            gts[image_name].path())
                    results[image_name].fns.append(gt_shape)
                    
            for pred_i, pred_shape in enumerate(pred_data['shapes']):
                if float(pred_shape['score']) < self.confidence_threshold:
                # if float(pred_shape['score']) < self.confs[pred_shape['label']]:
                    continue
                if pred_i not in pred_counted and self.is_in_roi(pred_shape):
                    match_gt = None
                    
                    for gt_i, gt_shape in enumerate(gt_data['shapes']):
                        if self.match_label(pred_shape, gt_shape) and (self.cal_iou(pred_shape, gt_shape) >= self.iou_threshold ):
                            match_gt = gt_shape
                            break
                    if match_gt == None:
                        # breakpoint()
                        self.fp += 1
                        if image_name not in results.keys():
                            results[image_name] = ImageResult(
                                gts[image_name].path())
                        results[image_name].fps.append(pred_shape)
                        
        # 情况三：pred里面所有的图片的名字，但是gt却没有这个图片
        for image_name in preds.keys():
            #所有的没被处理过的图片
            if image_name not in self.process_images:
                pred_data = preds[image_name].data()

                for pred_shape in pred_data['shapes']:
                    if float(pred_shape['score']) < self.confidence_threshold:
                    # if float(pred_shape['score']) < self.confs[pred_shape['label']]:
                        continue
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
        temp = {'jiaotongbiaozhi': {'fns': 0, 'fps': 0, 'tps': 0}}
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