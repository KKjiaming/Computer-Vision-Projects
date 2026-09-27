import yaml
from SMore_road.engine import RoadEngine

with open('./config/road.yaml') as yaml_f:
    config = yaml.load(yaml_f, Loader=yaml.FullLoader)
model_path = config['asphalt_cement_defect']['cement_defect']['model_path']
save_path = 'v0.5.0_auto'
d = RoadEngine()
d.defect_auto_path_run('/dataset/path/to/local_asset',
                       '/dataset/result/' + save_path + '/巡检测试集/')
