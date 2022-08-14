import matplotlib.pyplot as plt
import json

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch

matplotlib.use("Agg")


def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except:
        raise IOError("json file {} not found".format(file))


def save_json(data, file):
    try:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)
    except:
        raise IOError("json file {} write error".format(file))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class ResultGatherer(object):
    def __init__(self):
        self.reset()
        self.output = []
        self.target = []

    def reset(self):
        self.output = []
        self.target = []

    def update(self, output, target):
        self.output.append(output.detach().cpu().numpy().flatten())
        self.target.append(target.detach().cpu().numpy().flatten())

    def finalise(self):
        self.output = np.concatenate(self.output, axis=0)
        self.target = np.concatenate(self.target, axis=0)


class GtVsPredPlotter(ResultGatherer):
    def plot_gt_vs_pred(self):
        self.finalise()
        plt.scatter(self.target, self.output, color="g")

        min_val, max_val = min([np.min(self.target), np.min(self.output)]), max(
            [np.max(self.target), np.max(self.output)]
        )

        values = np.linspace(min_val, max_val, 9)
        plt.plot(values, values, "b--", linewidth=5)
        plt.xlabel("GT")
        plt.ylabel("Pred")
        plt.grid(True)

        self.reset()

        return plt.gcf()


def make_confusion_matrix_plot(output, target, conf_matrix_func, save_path):
    output = output.astype(np.int32)
    target = target.astype(np.int32)

    output = torch.from_numpy(output)
    target = torch.from_numpy(target)

    min_val, max_val = min(torch.min(output), torch.min(target)), max(
        torch.max(output), torch.max(target)
    )
    # for conf matrix func, target has to be non-negative
    output = output - min_val
    target = target - min_val
    conf_matrix = np.round(conf_matrix_func(output, target).numpy() * 100)

    vals = [x for x in range(min_val, max_val + 1, 1)]
    # print(vals)

    # following based on: https://stackoverflow.com/a/42265865/798093
    df_cm = pd.DataFrame(conf_matrix, vals, vals)
    plt.figure(figsize=(20, 17))
    plt.ticklabel_format(useOffset=False)

    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 24}, fmt="g")  # font size
    plt.xlabel("GT")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
