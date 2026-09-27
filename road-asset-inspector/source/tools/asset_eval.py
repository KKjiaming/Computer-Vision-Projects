from SMore_road.metric import AssetMetric

pr_metric = AssetMetric('./config/road.yaml')
pr_metric.run(vis_badcase=False, send_message=False, test_folder='demo')
