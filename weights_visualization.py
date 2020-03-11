import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.patches as mpatches


def _get_weights(filename):
    weights = np.zeros((10000, 2), dtype=np.float)
    with open(filename, "r", encoding="utf-8") as weightsFile:
        for i, weight_pair in enumerate(weightsFile):
            current_weights = weight_pair.replace(",", ".").split("\t")
            weights[i] = [float(current_weights[0]), float(current_weights[1])]
            # print(weights[i])
    return weights


def _get_compact_weights(filename, limit=None):
    weights = []
    frequency = []
    discrete_time = []
    with open(filename, "r", encoding="utf-8") as weights_file:
        for i, weights_triple in enumerate(weights_file):
            current_weights = weights_triple.replace(",", ".").split("\t")
            weights.append([current_weights[1], current_weights[2]])
            discrete_time_base = int(float(current_weights[0].strip()))
            discrete_time.append(discrete_time_base)
            curr_frequency = int(float(current_weights[2].strip()))
            frequency.append(curr_frequency)
            for k in range(0, curr_frequency):
                weights.append([current_weights[0], current_weights[1]])
                discrete_time_base += 1
                discrete_time.append(discrete_time_base)
            if limit is not None and (i == limit or discrete_time_base >= limit):
                print("Limit reached")
                break
    return discrete_time, weights


def _get_compact_discrete_weights(filename, limit=None):
    weights = []
    frequency = []
    discrete_time = []
    with open(filename, "r", encoding="utf-8") as weights_file:
        print(f"Reading file {filename}")
        for i, weights_triple in enumerate(weights_file):
            current_weights = weights_triple.replace(",", ".").split("\t")
            weights.append([int(current_weights[1]), int(current_weights[2])])
            discrete_time_base = int(current_weights[0].strip())
            discrete_time.append(discrete_time_base)
            curr_frequency = int(current_weights[3].strip())
            frequency.append(curr_frequency)
            for k in range(0, curr_frequency):
                weights.append([current_weights[1], current_weights[2]])
                discrete_time_base += 1
                discrete_time.append(discrete_time_base)
            if limit is not None and (i == limit or discrete_time_base >= limit):
                print("Limit reached")
                break
    return discrete_time, weights


def _plot_discrete_weights_compact_format(filename, anomalies=None, limit=None):
    _disc_time, _weights = _get_compact_discrete_weights(filename, limit)
    print("Num weights: ", len(_weights), " num timestamps: ", len(_disc_time))
    weights = np.asarray(_weights[:], dtype=np.uint32)
    labels = ['lru', 'lfu']
    shape = ['r-', 'b-']
    for col, lb, shape in zip(weights.T, labels, shape):
        plt.plot(_disc_time, col, shape, label=lb)

    if anomalies is not None:
        _anomaly_disc, _anomaly_weights = _get_discrete_anomalies(anomalies, limit)
        plt.plot(_anomaly_disc, _anomaly_weights, 'gD')
    plt.show()


def plot_weights(filename):
    weights = _get_weights(filename)
    plt.plot(weights)
    plt.show()


def plot_smooth_weights(filename):
    weights = _get_weights(filename)
    xnew = np.linspace(0.0, 1.0, 300)
    spl = make_interp_spline(weights, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(power_smooth)
    plt.show()


def _get_discrete_anomalies(anomalies, limit=None):
    discrete_time = []
    weights = []
    with open(anomalies, "r", encoding="utf-8") as anomaly_file:
        for i, weights_double in enumerate(anomaly_file):
            current_weights = weights_double.replace(",", ".").split("\t")
            curr_disc_time = int(current_weights[0].strip())
            discrete_time.append(curr_disc_time)
            weights.append(int(current_weights[1]))
            if limit is not None and (i == limit or curr_disc_time >= limit):
                print("Limit reached")
                break
    return discrete_time, np.asarray(weights, dtype=np.uint32)


def _get_anomalies(anomalies, limit=None):
    discrete_time = []
    weights = []
    with open(anomalies, "r", encoding="utf-8") as anomaly_file:
        for i, weights_double in enumerate(anomaly_file):
            current_weights = weights_double.replace(",", ".").split("\t")
            curr_disc_time = int(float(current_weights[0].strip()))
            discrete_time.append(curr_disc_time)
            weights.append(current_weights[1])
            if limit is not None and (i == limit or curr_disc_time >= limit):
                print("Limit reached")
                break
    return discrete_time, np.asarray(weights, dtype=np.float32)


def plot_weights_with_compact_format(filename, anomalies=None, limit=None, ):
    _disc_time, _weights = _get_compact_weights(filename, limit)
    print("Num weights: ", len(_weights), " num timestamps: ", len(_disc_time))
    weights = np.asarray(_weights[:], dtype=np.float32)
    labels = ['lru', 'lfu']
    shape = ['r-', 'b-']
    for col, lb, shape in zip(weights.T, labels, shape):
        plt.plot(_disc_time, col, shape, label=lb)

    if anomalies is not None:
        _anomaly_disc, _anomaly_weights = _get_anomalies(anomalies, limit)
        plt.plot(_anomaly_disc, _anomaly_weights, 'gD')
    plt.show()


def plot_several_weights_with_compact_format(filename_list, labels, limit=None):
    shape = ['r-', 'b-', 'g-', 'm-', 'c-', 'y-']
    index = 0
    plots = []
    for filename in filename_list:
        _disc_time, _weights = _get_compact_weights(filename, limit)
        print("Num weights: ", len(_weights), " num timestamps: ", len(_disc_time))
        weights = np.asarray(_weights[:], dtype=np.float32)

        for col, lb in zip(weights.T, labels):
            plot, = plt.plot(_disc_time, col, shape[index], label=lb)
            plots.append(plot)
            index += 1
    plt.legend(plots, labels)
    plt.show()


if __name__ == '__main__':
    basedir = "C:/Users/havar/Home/cache_simulation_results/"
    '''
    plot_weights_with_compact_format(
        filename=basedir + "weights_ex_1_lr03_dyn99.csv",
        anomalies=basedir + "anomalies_ex_1_lr03_dyn99.csv",
        limit=2000000
    )
    '''
    filenames = \
        [basedir + "res_LR_w.csv", basedir + "res_LR_W_w.csv", basedir + "res_None_w.csv"]
    # _plot_hit_rates([basedir + "res_LR_r.csv", basedir + "res_LR_W_r.csv", basedir + "res_None_r.csv"],
    #                ["LR", "LR_W", "None"], None)
    plot_several_weights_with_compact_format(filenames,
                                             ["LR_LRU", "LR_LFU", "LR_W_LRU", "LR_W_LFU", "None_LRU", "None_LFU"], 20000)
    '''
    _plot_discrete_weights_compact_format(
        filename=basedir + "scaled_5_w.csv",
        anomalies=basedir + "scaled_5_a.csv",
        limit=None
    )
    '''
