import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


def _get_weights(filename):
    weights = np.zeros((10000, 2), dtype=np.float)
    with open(filename, "r", encoding="utf-8") as weightsFile:
        for i, weight_pair in enumerate(weightsFile):
            current_weights = weight_pair.replace(",", ".").split("\t")
            weights[i] = [float(current_weights[0]), float(current_weights[1])]
            # print(weights[i])
    return weights


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


if __name__ == '__main__':
    plot_weights("C:/Users/havar/Home/caffeine/weights.csv")
