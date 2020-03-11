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


def _plot_hit_rates(filename_list, labels, limit=None):
    shape = ['r-', 'b-', 'g-', 'm-']
    index = 0
    plots = []
    for filename in filename_list:
        _disc_time, _hit_rates = _read_hit_rates(filename, limit)
        print("Num hit rates: ", len(_hit_rates), " num timestamps: ", len(_disc_time))

        disc_time = np.asarray(_disc_time, dtype=np.uint32)
        hit_rates = np.asarray(_hit_rates[:], dtype=np.float32)
        global_max_idx = np.argmax(hit_rates)
        global_max = hit_rates[global_max_idx]
        global_max_timestamp = disc_time[global_max_idx]
        print(f"{filename} is largest at {global_max_timestamp} with value {global_max}")
        _means = running_mean(hit_rates, 10)
        _mean_disc_time = running_mean(disc_time, 10)
        plot, = plt.plot(disc_time, hit_rates, shape[index], label=labels[index])
        plots.append(plot)
        index += 1
    plt.legend(plots, labels)
    plt.show()


def _read_hit_rates_flagged(filename, limit=None):
    weights = []
    discrete_time = []
    flagged = []
    with open(filename, "r", encoding="utf-8") as weights_file:
        for i, weights_triple in enumerate(weights_file):
            current_hit = weights_triple.replace(",", ".").split("\t")
            weights.append(float(current_hit[1].strip()))
            discrete_time.append(int(float(current_hit[0].strip())))
            flagged.append(bool(current_hit[2].strip()))
            if limit is not None and (i == limit or int(float(current_hit[0].strip())) >= limit):
                print("Limit reached")
                break
    return discrete_time, weights, flagged


def _plot_hit_rates_with_flagged(filename, limit=None):
    shape = ['r-', 'b-', 'g-', 'm-']
    index = 0
    _disc_time, _hit_rates, _flagged = _read_hit_rates_flagged(filename, limit)
    print("Num hit rates: ", len(_hit_rates), " num timestamps: ", len(_disc_time), " num flagged: ",
          np.count_nonzero(_flagged))

    disc_time = np.asarray(_disc_time, dtype=np.uint32)
    hit_rates = np.asarray(_hit_rates[:], dtype=np.float32)
    plt.scatter(disc_time, hit_rates, c=_flagged, marker="o", cmap="bwr_r")
    plt.show()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == '__main__':
    basedir = "C:/Users/havar/Home/cache_simulation_results/"

    # _plot_hit_rates_with_flagged(basedir + "res_01_r.csv")
    _plot_hit_rates([basedir + "res_LR_r.csv", basedir + "res_LR_W_r.csv", basedir + "res_None_r.csv", basedir + "arc_r.csv"],
                    ["LR", "LR_W", "None", "ARC"], None)
    # _plot_hit_rates([basedir + "res_None_r.csv"], None)

    mylist = [1, 2, 3, 4, 5, 6, 7]
    print(running_mean(np.asarray(mylist), 4))

    '''
        colors = [0,0,0,1,0,1] #red is 0, blue is 1
        ax.scatter(data[:,0],data[:,1],c=colors,marker="o", cmap="bwr_r")
    '''
