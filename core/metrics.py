import torchmetrics


def get_metric(metric_name, num_classes=10):
    metricname_to_func = {
        "L2": torchmetrics.MeanSquaredError(),
        "L1": torchmetrics.MeanAbsoluteError(),
        "CONF": torchmetrics.ConfusionMatrix(num_classes=num_classes, normalize="true"),
    }
    return metricname_to_func[metric_name]
