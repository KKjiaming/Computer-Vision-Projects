from SMore_road.metric import AsphaltMetric

pr_metric = AsphaltMetric('./config/road.yaml')
pr_metric.run(vis_badcase=False, send_message=False, test_folder='沥青-9.15山西太原')
pr_metric.run(
    vis_badcase=False, send_message=False, test_folder='沥青-11.14沿江高速')
pr_metric.run(
    vis_badcase=False, send_message=False, test_folder='沥青-11.15广深高速')
pr_metric.run(
    vis_badcase=False, send_message=False, test_folder='沥青-11.15沿江高速')
pr_metric.run(
    vis_badcase=False, send_message=False, test_folder='沥青-11.17广深高速')
