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


def _get_compact_weights(filename):
    weights = []
    frequency = []
    discrete_time = []
    with open(filename, "r", encoding="utf-8") as weights_file:
        for i, weights_triple in enumerate(weights_file):
            current_weights = weights_triple.replace(",", ".").split("\t")
            weights.append([current_weights[1], current_weights[2]])
            discrete_time.append(int(float(current_weights[0].strip())))
            curr_frequency = int(float(current_weights[2].strip()))
            frequency.append(curr_frequency)
            for k in range(0, curr_frequency):
                weights.append([current_weights[0], current_weights[1]])
    return discrete_time, weights


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


def plot_weights_with_compact_format(filename):
    _disc_time, _weights = _get_compact_weights(filename)
    print("Num sampes: ", len(_weights))
    weights = np.asarray(_weights[:], dtype=np.float32)
    labels = ['lru', 'lfu']
    shape = ['r-', 'b-']
    for col, lb, shape in zip(weights.T, labels, shape):
        plot = plt.plot(_disc_time, col, shape, label=lb)
        # plt.legend(handler_map={plot: HandlerLine2D(numpoints=4)})

    plt.show()


if __name__ == '__main__':
    plot_weights_with_compact_format("C:/Users/havar/Home/cache_simulation_results/weights13.csv")
