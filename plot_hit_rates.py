import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline


def _read_hit_rates(filename, limit=None):
    weights = []
    discrete_time = []
    with open(filename, "r", encoding="utf-8") as weights_file:
        for i, weights_triple in enumerate(weights_file):
            current_hit = weights_triple.replace(",", ".").split("\t")
            weights.append(float(current_hit[1].strip()))
            discrete_time.append(int(float(current_hit[0].strip())))
            if limit is not None and (i == limit or int(float(current_hit[0].strip())) >= limit):
                print("Limit reached")
                break
    return discrete_time, weights


def _plot_hit_rates(filename_list, limit=None):
    shape = ['r-', 'b-', 'g-', 'm-']
    index = 0
    for filename in filename_list:
        _disc_time, _hit_rates = _read_hit_rates(filename, limit)
        print("Num hit rates: ", len(_hit_rates), " num timestamps: ", len(_disc_time))
        disc_time = np.asarray(_disc_time, dtype=np.uint32)
        hit_rates = np.asarray(_hit_rates[:], dtype=np.float32)
        plt.plot(disc_time, hit_rates)
    plt.show()


if __name__ == '__main__':
    basedir = "C:/Users/havar/Home/cache_simulation_results/"

    _plot_hit_rates([basedir + "scaled_6_r.csv"])
