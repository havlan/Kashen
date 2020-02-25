from scipy.interpolate import interp1d
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

from plot_hit_rates import _read_hit_rates, _plot_hit_rates


def _make_delta_hit_rates(filename, limit=None):
    weights = []
    discrete_time = []
    prev_weight = 0.0
    with open(filename, "r", encoding="utf-8") as weights_file:
        for i, weights_triple in enumerate(weights_file):
            current_hit = weights_triple.replace(",", ".").split("\t")
            discrete_time.append(int(float(current_hit[0].strip())))
            current_weight = float(current_hit[1].strip())
            current_delta = current_weight - prev_weight
            weights.append(current_delta)
            prev_weight = current_weight
            if limit is not None and (i == limit or int(float(current_hit[0].strip())) >= limit):
                print("Limit reached")
                break

    return discrete_time, weights


def _plot_deltas(filename, limit=None):
    _t, _w = _make_delta_hit_rates(filename, limit)
    disc_time = np.asarray(_t, dtype=np.uint32)
    hit_rates = np.asarray(_w[:], dtype=np.float32)
    plt.plot(disc_time, hit_rates)
    f1 = interp1d(disc_time, hit_rates, kind="cubic")
    plt.plot(disc_time, f1(disc_time), "--")

    plt.show()


def _plot_hit_rates(filename, limit=None):
    shape = ['r-', 'b-']
    index = 0
    _disc_time, _hit_rates = _read_hit_rates(filename, limit)
    print("Num hit rates: ", len(_hit_rates), " num timestamps: ", len(_disc_time))
    disc_time = np.asarray(_disc_time, dtype=np.uint32)
    hit_rates = np.asarray(_hit_rates[:], dtype=np.float32)
    plt.plot(disc_time, hit_rates)
    plt.show()


def _get_local_minimas(filename, limit=None):
    _t, _w = _read_hit_rates(filename, limit)
    _plot_hit_rates(filename, limit)
    print(np.var(_w))
    print(np.min(_w))
    print(np.argmin(_w))


if __name__ == '__main__':
    basedir = "C:/Users/havar/Home/cache_simulation_results/"
    curr_filename = basedir + "res_01_r.csv"

    # _plot_deltas(curr_filename, 400000)
    _get_local_minimas(curr_filename)
