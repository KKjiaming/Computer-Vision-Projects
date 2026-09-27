import os

import yaml
from SMore_road.engine import RoadEngine

with open('./config/road.yaml') as yaml_f:
    config = yaml.load(yaml_f, Loader=yaml.FullLoader)
model_path = config['asphalt_cement_defect']['cement_defect']['model_path']
save_path = os.path.basename(model_path).replace('.onnx', '')
d = RoadEngine()
d.cement_defect_path_run('/dataset/path/to/local_asset',
                         '/dataset/result/' + save_path + '/水泥-benchmark/')
