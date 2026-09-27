import os

import yaml
from SMore_road.engine import RoadEngine

with open('./config/road.yaml') as yaml_f:
    config = yaml.load(yaml_f, Loader=yaml.FullLoader)
model_path = config['asphalt_cement_defect']['asphalt_defect']['model_path']
save_path = os.path.basename(model_path).replace('.onnx', '')
d = RoadEngine()

d.asphalt_defect_path_run('/dataset/path/to/local_asset',
                          '/dataset/result/' + save_path + '/沥青-9.15山西太原/')

d.asphalt_defect_path_run('/dataset/path/to/local_asset',
                          '/dataset/result/' + save_path + '/沥青-11.14沿江高速/')

d.asphalt_defect_path_run('/dataset/path/to/local_asset',
                          '/dataset/result/' + save_path + '/沥青-11.15广深高速/')

d.asphalt_defect_path_run('/dataset/path/to/local_asset',
                          '/dataset/result/' + save_path + '/沥青-11.15沿江高速/')

d.asphalt_defect_path_run('/dataset/path/to/local_asset',
                          '/dataset/result/' + save_path + '/沥青-11.17广深高速/')
