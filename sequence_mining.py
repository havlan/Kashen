from prefixspan import PrefixSpan
from weights_visualization import _get_compact_discrete_weights


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _read_db(filename, limit=None):
    weights = []
    frequency = []
    discrete_time = []
    with open(filename, "r", encoding="utf-8") as weights_file:
        print(f"Reading file {filename}")
        for i, weights_triple in enumerate(weights_file):
            current_weights = weights_triple.replace(",", ".").split("\t")
            weights.append(int(current_weights[1]))
            discrete_time_base = int(current_weights[0].strip())
            discrete_time.append(discrete_time_base)
            curr_frequency = int(current_weights[3].strip())
            frequency.append(curr_frequency)
            for k in range(0, curr_frequency):
                weights.append(int(current_weights[1]))
                discrete_time_base += 1
                discrete_time.append(discrete_time_base)
            if limit is not None and (i == limit or discrete_time_base >= limit):
                print("Limit reached")
                break
    return discrete_time, weights


if __name__ == '__main__':
    basedir = "C:/Users/havar/Home/cache_simulation_results/"

    _t, _w = _read_db(basedir + "scaled_w_01.csv")
    data = list(chunks(_w, 1000))
    ps = PrefixSpan(data)
    ps.minlen = 5
    ps.maxlen = 100

    print(ps.frequent(5, closed=True))
