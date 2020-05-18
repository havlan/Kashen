import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from paretoset import paretoset

dirname = os.path.dirname(__file__)

def get_eviction_df_from_file(filename, policy, take_prefix):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1, 8]).sort_values(by=["Policy"])
    relative_hit_rate = []
    relative_percentage_hit_rate = []
    relative_weighted_hit_rate = []
    relative_percentage_weighted_hit_rate = []
    data["Weighted hit rate"] = data["Weighted hit rate"].str.replace(",", ".").astype(float)
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)
    data = data[~data["Policy"].str.contains("_")]
    if take_prefix:
        data = data[data["Policy"].str.contains(policy.split(".")[0])]
    else:
        data = data[data["Policy"].str.contains(policy)]
    base = data.loc[data['Policy'] == policy]
    print("BASE: ", base.to_string())
    for index, row in data.iterrows():
        base_hit_rate = round(float(base["Hit rate"]), 2)
        base_weighted_hit_rate = round(float(base["Weighted hit rate"]), 2)
        print(f"Comparing base {base['Policy']} with row {row['Policy']}")
        hit_rate = round(float(row["Hit rate"]) - float(base["Hit rate"]), 2)
        if base_hit_rate == 0.0:
            pct_hit_rate = 0.0
        else:
            pct_hit_rate = round(100.0 * (hit_rate / float(base["Hit rate"])), 2)
        relative_hit_rate.append(hit_rate)
        relative_percentage_hit_rate.append(pct_hit_rate)

        weighted_hit_rate = round(float(row["Weighted hit rate"]) - float(base["Weighted hit rate"]), 2)
        if base_weighted_hit_rate == 0.0:
            pct_weighted_hit_rate = 0.0
        else:
            pct_weighted_hit_rate = round(100.0 * (weighted_hit_rate / float(base["Weighted hit rate"])), 2)
        relative_weighted_hit_rate.append(weighted_hit_rate)
        relative_percentage_weighted_hit_rate.append(pct_weighted_hit_rate)

    # relative_percentage_hit_rate = [str(f"{el}%") for el in relative_percentage_hit_rate]
    # relative_percentage_weighted_hit_rate = [str(f"{el}%") for el in relative_percentage_weighted_hit_rate]

    # data["Hit rate change"] = relative_hit_rate
    data["Percentage hit rate change"] = relative_percentage_hit_rate
    # data["Weighted hit rate change"] = relative_weighted_hit_rate
    data["Percentage weighted hit rate change"] = relative_percentage_weighted_hit_rate
    data.to_csv(os.path.join(dirname + filename + "_analyzed_hello.txt"))

    return data.drop(["Hit rate", "Weighted hit rate"], axis=1)

#
def get_admittor_df_from_file(filename):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1, 8]).sort_values(by=["Policy"])
    # data = data.sort_values(by=["Policy"])
    relative_hit_rate = []
    relative_percentage_hit_rate = []
    relative_weighted_hit_rate = []
    relative_percentage_weighted_hit_rate = []
    data["Weighted hit rate"] = data["Weighted hit rate"].str.replace(",", ".").astype(float)
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)
    base = data.iloc[0]
    print("BASE: ", base.to_string())
    for index, row in data.iterrows():
        # print(index)
        if '_' not in row["Policy"]:
            base = row

        base_hit_rate = round(float(base["Hit rate"]), 2)
        base_weighted_hit_rate = round(float(base["Weighted hit rate"]), 2)
        print(f"Comparing base {base['Policy']} with row {row['Policy']}")
        hit_rate = round(float(row["Hit rate"]) - float(base["Hit rate"]), 2)
        if base_hit_rate == 0.0:
            pct_hit_rate = 0.0
        else:
            pct_hit_rate = round(100.0 * (hit_rate / float(base["Hit rate"])), 2)
        relative_hit_rate.append(hit_rate)
        relative_percentage_hit_rate.append(pct_hit_rate)

        weighted_hit_rate = round(float(row["Weighted hit rate"]) - float(base["Weighted hit rate"]), 2)
        if base_weighted_hit_rate == 0.0:
            pct_weighted_hit_rate = 0.0
        else:
            pct_weighted_hit_rate = round(100.0 * (weighted_hit_rate / float(base["Weighted hit rate"])), 2)
        relative_weighted_hit_rate.append(weighted_hit_rate)
        relative_percentage_weighted_hit_rate.append(pct_weighted_hit_rate)

    # relative_percentage_hit_rate = [str(f"{el}%") for el in relative_percentage_hit_rate]
    # relative_percentage_weighted_hit_rate = [str(f"{el}%") for el in relative_percentage_weighted_hit_rate]

    # data["Hit rate change"] = relative_hit_rate
    data["Percentage hit rate change"] = relative_percentage_hit_rate
    # data["Weighted hit rate change"] = relative_weighted_hit_rate
    data["Percentage weighted hit rate change"] = relative_percentage_weighted_hit_rate
    data.to_csv(os.path.join(dirname + filename + "_analyzed_hello.txt"))

    return data.drop(["Hit rate", "Weighted hit rate"], axis=1)


def analyze_directory(directory_name):
    directory_name_stripped = directory_name.split("/")[-1]
    dfs = []
    for file in os.listdir(os.path.join(dirname + directory_name)):
        if len(file.split("_")) == 2:
            print(file)
            dfs.append(get_admittor_df_from_file(directory_name + "/" + file))
    merged = pd.concat(dfs, axis=0)
    merged = merged.groupby("Policy").mean()
    # print(merged.index)
    merged.to_csv(
        os.path.join(dirname + directory_name + f"/{directory_name_stripped}_per_policy_pcentage_analysis.txt"))
    admittors = []
    for idx, row in merged.iterrows():
        admittor = ""
        if "_" in idx:
            admittor = idx.split("_")[1]
        else:
            admittor = "None"
        admittors.append(admittor)
    merged["Admittor"] = admittors
    print(merged)

    merged = merged.groupby("Admittor").mean()
    merged.to_csv(
        os.path.join(dirname + directory_name + f"/{directory_name_stripped}_per_admittor_pcentage_analysis.txt"))
    print(merged)


def analyze_simulation(sim_filename, exclude_policies=None):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    data = get_admittor_df_from_file(filename=sim_filename)
    data = data.groupby("Policy").mean()
    if exclude_policies is not None:
        for p in exclude_policies:
            print(f"Excluding {p}")
            data = data[~data.index.str.contains(p)]
    print(f"After filter {data.head(100)}")
    admittors = []
    for idx, row in data.iterrows():
        if "_" in idx:
            admittor = idx.split("_")[1]
        else:
            admittor = "None"
        admittors.append(admittor)
    data["Admittor"] = admittors
    data = data.groupby("Admittor").mean()
    print(data)
    data["Percentage hit rate change"] = data["Percentage hit rate change"].map('{:,.2f}%'.format)
    data["Percentage weighted hit rate change"] = data["Percentage weighted hit rate change"].map('{:,.2f}%'.format)
    print (data)

def analyze_eviction_simulation(sim_filename, policy="heap.None", take_prefix=True):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    data = get_eviction_df_from_file(filename=sim_filename, policy=policy, take_prefix=take_prefix)
    data = data.groupby("Policy").mean()
    data["Percentage hit rate change"] = data["Percentage hit rate change"].map('{:,.2f}%'.format)
    data["Percentage weighted hit rate change"] = data["Percentage weighted hit rate change"].map('{:,.2f}%'.format)
    print (data)

def plot_df(data):
    sns.set_context('paper')
    sns.set(style='ticks')
    print("PLOT_DF", data.head(1))
    # fg = sns.FacetGrid(data=data, aspect=1.66)
    # fg.map(plt.scatter, 'Percentage hit rate change')
    fg = sns.barplot(data.index, data['Percentage hit rate change'], alpha=0.8)
    fg.set_xlabel("Admittor")
    fg.set_ylabel("Percentage hit rate change")
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # analyze_directory("/results/web")
    analyze_simulation("/results/financial/financial1.trace.txt")#, ["Mru", "Mfu"])
    #analyze_eviction_simulation("/results/web/web_0.txt")#,["Mru", "Mfu"])

'''
Comparison                      -25.86%                             -35.35%
None                              0.00%                               0.00%
Secondary                        -8.85%                             -30.97%
Threshold15                       1.03%                              -2.01%
TinyLfu                          35.93%                              15.55%
TinyLfuBoost                     27.35%                               6.64%
TinyLfuMulti                    -57.59%                             -64.25%
'''