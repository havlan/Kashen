import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from decimal import Decimal

dirname = os.path.dirname(__file__)

sns.set_context('paper')
sns.set(style='ticks')


def filter_out_policies_based_on_prefix(data, policies):
    if policies is None:
        return data

    tmp = data
    for p in policies:
        tmp = tmp[~tmp.index.str.contains(p)]
    return tmp
    # return data[~(data.index.str in policies)]
    # return data[(data.index.str.contains("Comparison"))]


def filter_out_policies_based_on_name(data, policies):
    if policies is None:
        return data

    tmp = data
    for p in policies:
        tmp = tmp[~tmp.index.str.contains(p)]

    return tmp


def scatter_plot_based_on_policy(filename):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1, 8]).set_index("Policy")
    data = filter_out_policies_based_on_prefix(data, ["sampled", "product", "heap"])
    data["Weighted Hit Rate"] = data["Weighted Hit Rate"].str.replace(",", ".").astype(float)
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)
    admittormap = {
        "linked.Fifo": "r",
        "linked.Lru": "g",
        "linked.Clock": "b",
        "sampled.Random": "c",
        "sampled.Lfu": "m",
        "sampled.Lru": "y",
        "sampled.Hyperbolic": "k",
        "sampled.Mfu": "darkorchid",
        "product.Caffeine": "springgreen",
        "sampled.Mru": "lightsalmon",
        "linked.Lfucostboost": "grey",
        "heap.": ""
    }
    policies = list(data.index)
    policies_found = []
    for p in policies:
        if '_' in p:
            found = p.split("_")[0]
        else:
            found = p
        policies_found.append(found)
    data["Eviction"] = policies_found
    print(policies_found)
    fg = sns.FacetGrid(data=data, hue='Eviction', aspect=1.66)
    fg.map(plt.scatter, 'Weighted Hit Rate', 'Hit rate')
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def scatter_plot_based_on_admittor(filename, filter_policies_prefix=None, filter_policies_name=None):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1, 8]).set_index("Policy")
    data = filter_out_policies_based_on_prefix(data, filter_policies_prefix)
    data = filter_out_policies_based_on_name(data, filter_policies_name)
    data["Weighted Hit Rate"] = data["Weighted Hit Rate"].str.replace(",", ".").astype(float)
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)

    print(data)
    color_vector = list(data.index)
    print(color_vector)
    admittormap = {
        "None": "r",
        "Comparison": "g",
        "Secondary": "b",
        "Threshold15": "c",
        "TinyLfu": "m",
        "TinyLfuBoost": "y",
        "TinyLfuMulti": "k"
    }
    color_list = []
    admittors = []
    for s in color_vector:
        if '_' in s:
            found = s.split("_")[1]
        else:
            found = "None"
        admittors.append(found)
        color_list.append(admittormap[found])
    data["Admittor"] = admittors
    data["Colors"] = color_list
    admittorset = set(admittors)
    print(data)

    fg = sns.FacetGrid(data=data, hue='Admittor', aspect=1.66)
    # box = fg.get_position()
    # fg.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    fg.map(plt.scatter, 'Weighted Hit Rate', 'Hit rate')
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def get_change_matrix_with_admission(filename):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1, 8]).set_index("Policy")
    data = data.sort_values(by=["Policy"])
    relative_hit_rate = []
    relative_percentage_hit_rate = []
    relative_weighted_hit_rate = []
    relative_percentage_weighted_hit_rate = []
    data["Weighted Hit Rate"] = data["Weighted Hit Rate"].str.replace(",", ".").astype(float)
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)

    # remove product and sampled
    data = data[~data.index.str.contains("product")]
    data = data[~data.index.str.contains("sampled")]

    # only comparison admittor
    data = data[(data.index.str.contains("Comparison")) | (~data.index.str.contains('_'))]
    print("DATA AFTER FILTERING: ", data)

    base = data.iloc[0]
    for index, row in data.iterrows():
        # print(index)
        if '_' not in index:
            base = row

        hit_rate = round(float(row["Hit rate"]) - float(base["Hit rate"]), 2)
        pct_hit_rate = round(100.0 * (hit_rate / float(base["Hit rate"])), 2)
        relative_hit_rate.append(hit_rate)
        relative_percentage_hit_rate.append(pct_hit_rate)

        weighted_hit_rate = round(float(row["Weighted Hit Rate"]) - float(base["Weighted Hit Rate"]), 2)
        pct_weighted_hit_rate = round(100.0 * (weighted_hit_rate / float(base["Weighted Hit Rate"])), 2)
        relative_weighted_hit_rate.append(weighted_hit_rate)
        relative_percentage_weighted_hit_rate.append(pct_weighted_hit_rate)

    relative_percentage_hit_rate = [str(f"{el}%") for el in relative_percentage_hit_rate]
    relative_percentage_weighted_hit_rate = [str(f"{el}%") for el in relative_percentage_weighted_hit_rate]

    data["Hit rate change"] = relative_hit_rate
    data["Percentage hit rate change"] = relative_percentage_hit_rate
    data["Weighted hit rate change"] = relative_weighted_hit_rate
    data["Percentage weighted hit rate change"] = relative_percentage_weighted_hit_rate
    # data["Relative weighted hit rate"]

    print(data)
    data.to_csv(os.path.join(dirname + filename + "comparison_relative_change.txt"))


def admission_bar_plot(filename, filter_policies_prefix=None):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1]).set_index("Policy")
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)
    data = filter_out_policies_based_on_prefix(data, filter_policies_prefix)
    policy_index = data.index
    admittors = []
    policies = []
    for s in policy_index:
        tmp = s.split(".")[1]
        tmp = tmp.split("_")[0]
        policies.append(tmp)

        if '_' in s:
            admittor_found = s.split("_")[1]
        else:
            admittor_found = "None"
        admittors.append(admittor_found)
    data["Admittor"] = admittors
    data["Policy name"] = policies
    print(data)
    print("HEAD: \n", data.head())
    fg = sns.barplot(x="Policy name", y="Hit rate", hue="Admittor", data=data)
    #plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # scatter_plot_based_on_admittor("/results/web/web_0.txt", ["sampled", "product", "heap"])#, ["linked.Mru", "linked.Mfu"])
    admission_bar_plot("/results/web/web_0.txt",
                       ["sampled", "product", "heap", "linked.Lfucostboost", "linked.Clock"])
