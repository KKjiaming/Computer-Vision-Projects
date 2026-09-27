import json
import logging
import os
import sys

import cv2
import yaml

from ..module import AsphaltCementClassify, AsphaltDefect, Asset, CementDefect

IMG_FORMATS = [
    'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'
]


class RoadEngine(object):

    def __init__(self, config_path='./config/road.yaml'):
        with open(config_path) as yaml_f:
            self.config = yaml.load(yaml_f, Loader=yaml.FullLoader)
        self.datasets = []
        self.logger = logging.getLogger('RoadEngine')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '|%(asctime)s-%(name)s-%(levelname)s|: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.defect_config = self.config['asphalt_cement_defect']
        self.asphalt_cement_classify = AsphaltCementClassify(
            self.defect_config['asphalt_cement_classify'])
        self.asphalt_defect = AsphaltDefect(
            self.defect_config['asphalt_defect'])
        self.cement_defect = CementDefect(self.defect_config['cement_defect'])
        self.asset_config = self.config['asset']
        self.asset = Asset(self.asset_config)

    def generate_dataset(self):
        g = os.walk(self.input_path)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.split('.')[-1].lower() in IMG_FORMATS:
                    self.datasets.append(os.path.join(path, file_name))
        self.logger.info('Find ' + str(len(self.datasets)) +
                         ' images in Path: ' + self.input_path)

    # 交通资产直接跑文件夹
    def asset_path_run(self, input_path, output_path):
        self.datasets = []
        self.input_path = input_path
        assert os.path.exists(
            self.input_path), self.input_path + ' path not exist!!'
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.generate_dataset()
    
        # 图片按照顺序排序
        datset_dir = {}
        for path in self.datasets:
            # key = int(path.split('/')[-1].split('.')[0])
            # key = int(path.split('/')[-1].split('.')[0].split('_')[0])
            key = int(path.split('/')[-1].replace(".jpg","")[-4::])
            # breakpoint()
            value = path
            datset_dir.update({key:value})
    
        sorted_datset_dir = sorted(datset_dir.keys())
        sorted_dataset = [datset_dir[key] for key in sorted_datset_dir]
        self.datasets = sorted_dataset
        
        for image_path in self.datasets:
            self.logger.info('process ' + image_path)
            image = cv2.imread(image_path)
            if image is None:
                continue
            output = self.asset_run(image)
            self.write_asset_json(output, image_path)

    # 交通资产对外接口
    def asset_run(self, image):
        out = self.asset.run(image)
        return out

    # 自动模式直接跑文件夹
    def defect_auto_path_run(self, input_path, output_path):
        self.datasets = []
        self.input_path = input_path
        assert os.path.exists(
            self.input_path), self.input_path + ' path not exist!!'
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.generate_dataset()
        for image_path in self.datasets:
            self.logger.info('process ' + image_path)
            image = cv2.imread(image_path)
            if image is None:
                continue
            output, road_type = self.defect_auto_run(
                image, confidence_threshold=0.1)
            self.write_json(output, image_path, road_type=road_type)

    # 自动模式对外接口
    def defect_auto_run(self, image, confidence_threshold=None):
        road_type = self.asphalt_cement_classify.run(image)
        if road_type == 'asphalt':
            return self.asphalt_defect_run(
                image, confidence_threshold=confidence_threshold), road_type
        if road_type == 'cement':
            return self.cement_defect_run(
                image, confidence_threshold=confidence_threshold), road_type

    # 沥青直接跑文件夹
    def asphalt_defect_path_run(self, input_path, output_path):
        self.datasets = []
        self.input_path = input_path
        assert os.path.exists(
            self.input_path), self.input_path + ' path not exist!!'
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.generate_dataset()
        for image_path in self.datasets:
            self.logger.info('process ' + image_path)
            image = cv2.imread(image_path)
            if image is None:
                continue
            output = self.asphalt_defect_run(image, confidence_threshold=0.1)
            self.write_json(output, image_path)

    # 沥青对外接口
    def asphalt_defect_run(self, image, confidence_threshold=None):
        return self.asphalt_defect.run(
            image, confidence_threshold=confidence_threshold)

    # 水泥直接跑文件夹
    def cement_defect_path_run(self, input_path, output_path):
        self.datasets = []
        self.input_path = input_path
        assert os.path.exists(
            self.input_path), self.input_path + ' path not exist!!'
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.generate_dataset()
        for image_path in self.datasets:
            self.logger.info('process ' + image_path)
            image = cv2.imread(image_path)
            if image is None:
                continue
            output = self.cement_defect_run(image, confidence_threshold=0.1)
            self.write_json(output, image_path)

    # 水泥对外接口
    def cement_defect_run(self, image, confidence_threshold=None):
        return self.cement_defect.run(
            image, confidence_threshold=confidence_threshold)
        
    def write_asset_json(self, output, image_path, road_type=None, vis=True):
        json_data = {
            'version': '1.0.1',
            'flags': {},
            'shapes': [],
            'imagePath': os.path.basename(image_path),
            'imageData': None,
            'imageHeight': 2056,
            'imageWidth': 2464
        }
        if vis:
            image = cv2.imread(image_path)
            if road_type is not None:
                image = cv2.putText(image, road_type, (int(100), int(100)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                                    1)
        has_defect = False
        # boxes_, scores_, labels_, ocr_names_, retrival_names_, track_id_
        boxes, scores, labels, is_covereds, ocr_names, retrival_names,track_ids_ = output
        shapes = []
        # breakpoint()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            cls_name = labels[i]
            is_covered = is_covereds[i]
            ocr_name = ocr_names[i]
            retrival_name = retrival_names[i]
            # 交通资产类别独有
            track_id = track_ids_[i]
            s = {
                'label': cls_name,
                'points': [[float(x1), float(y1)], [float(x2),
                                                    float(y2)]],
                'score': float(score),
                'ocr_name':ocr_name,
                'is_covered':is_covered,
                'retrival_name':retrival_name,
                'track_id':track_id, 
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {}
            }
            shapes.append(s)
            # if is_covered == True:
            has_defect = True
            
            if cls_name is None:
                # 名字是空的，只输出id
                id_name = str(track_id)
            else:
                if track_id is not None:
                    id_name = str(track_id)+"|"+cls_name
                # 名字存在，id非空就输出id+名字
                else:
                    continue
            
            if vis:
                image = cv2.rectangle(image, (int(x1), int(y1)),
                                      (int(x2), int(y2)),self.getcolor(track_id), 2)
                
                image = cv2.putText(image, id_name, (int(x1), int(y1)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1,self.getcolor(track_id),
                                    1)
                
                # image = cv2.putText(image, str(trackid), (int(x2), int(y2)),
                #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                #                     1)
        save_path = self.output_path + '/' + image_path[len(self.input_path):]
        if vis:
            cv2.imwrite(save_path, image)
        json_data['shapes'] = shapes
        if has_defect:
            # breakpoint()
            json_save_path = save_path.replace('.jpg', '.json')
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            self.logger.info("json_save_path" + json_save_path)
            with open(json_save_path, 'w') as f:
                f.write(json.dumps(json_data, indent=4))

        
    def getcolor(self, track_id):
        track_id = int(track_id) if track_id is not None else 0
        track_id += 100
        return (37 * track_id % 255, 17 * (track_id * 2) % 255, 29 * (track_id * 4) % 255 )
        
    def write_json(self, output, image_path, road_type=None, vis=True):
        json_data = {
            'version': '1.0.1',
            'flags': {},
            'shapes': [],
            'imagePath': os.path.basename(image_path),
            'imageData': None,
            'imageHeight': 2056,
            'imageWidth': 2464
        }
        if vis:
            image = cv2.imread(image_path)
            if road_type is not None:
                image = cv2.putText(image, road_type, (int(100), int(100)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                                    1)
        has_defect = False
        boxes, scores, labels = output
        shapes = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            cls_name = labels[i]
            s = {
                'label': cls_name,
                'points': [[float(x1), float(y1)], [float(x2),
                                                    float(y2)]],
                'score': float(score),
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {},
            }
            shapes.append(s)
            has_defect = True
            if vis:
                image = cv2.rectangle(image, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (0, 255, 0), 2)
                image = cv2.putText(image, cls_name, (int(x2), int(y2)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                                    1)
        save_path = self.output_path + '/' + image_path[len(self.input_path):]
        if vis:
            cv2.imwrite(save_path, image)
        json_data['shapes'] = shapes
        if has_defect:
            json_save_path = save_path.replace('.jpg', '.json')
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            # self.logger.info("json_save_path" + json_save_path)
            with open(json_save_path, 'w') as f:
                f.write(json.dumps(json_data, indent=4))
