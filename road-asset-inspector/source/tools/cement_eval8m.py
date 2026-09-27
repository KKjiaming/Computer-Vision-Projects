from SMore_road.metric import CementMetric

pr_metric = CementMetric('./config/road.yaml')
# pr_metric.all_run(vis_badcase=False, send_message=True)
pr_metric.run(
    vis_badcase=False, send_message=False, test_folder='水泥-benchmark')
