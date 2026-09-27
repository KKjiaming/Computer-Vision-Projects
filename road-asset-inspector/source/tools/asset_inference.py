import os

import yaml
from SMore_road.engine import RoadEngine

with open('./config/road.yaml') as yaml_f:
    config = yaml.load(yaml_f, Loader=yaml.FullLoader)
model_path = config['asset']['asset_detect']['detect_model_path']
save_path = os.path.basename(model_path).replace('.onnx', '')
d = RoadEngine()

# d.asset_path_run('/dataset/path/to/local_asset',
#                  '/path/to/local_asset' + save_path + '/水泥-01.04广西广昆高速')
#/dataset/path/to/local_asset

# d.asset_path_run('/dataset/path/to/local_asset',
#                 '/path/to/local_asset' + save_path + '/2023-02-21-15-41-29')
# /path/to/local_asset
d.asset_path_run('/workspace/test_out/zzn_zhedang/zhedang',
                '/path/to/local_asset' + save_path + '/zhedang')