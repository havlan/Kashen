import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)


#
def get_admittor_df_from_file(filename):
    data = pd.read_csv(os.path.join(dirname + filename), usecols=[0, 1, 8]).sort_values(by=["Policy"])
    print(data.head(129))
    # data = data.sort_values(by=["Policy"])
    relative_hit_rate = []
    relative_percentage_hit_rate = []
    relative_weighted_hit_rate = []
    relative_percentage_weighted_hit_rate = []
    data["Weighted Hit Rate"] = data["Weighted Hit Rate"].str.replace(",", ".").astype(float)
    data["Hit rate"] = data["Hit rate"].str.replace(",", ".").astype(float)
    base = data.iloc[0]
    for index, row in data.iterrows():
        # print(index)
        if '_' not in row["Policy"]:
            base = row

        base_hit_rate = round(float(base["Hit rate"]), 2)
        base_weighted_hit_rate = round(float(base["Weighted Hit Rate"]), 2)

        hit_rate = round(float(row["Hit rate"]) - float(base["Hit rate"]), 2)
        if base_hit_rate == 0.0:
            pct_hit_rate = 0.0
        else:
            pct_hit_rate = round(100.0 * (hit_rate / float(base["Hit rate"])), 2)
        relative_hit_rate.append(hit_rate)
        relative_percentage_hit_rate.append(pct_hit_rate)

        weighted_hit_rate = round(float(row["Weighted Hit Rate"]) - float(base["Weighted Hit Rate"]), 2)
        if base_weighted_hit_rate == 0.0:
            pct_weighted_hit_rate = 0.0
        else:
            pct_weighted_hit_rate = round(100.0 * (weighted_hit_rate / float(base["Weighted Hit Rate"])), 2)
        relative_weighted_hit_rate.append(weighted_hit_rate)
        relative_percentage_weighted_hit_rate.append(pct_weighted_hit_rate)

    # relative_percentage_hit_rate = [str(f"{el}%") for el in relative_percentage_hit_rate]
    # relative_percentage_weighted_hit_rate = [str(f"{el}%") for el in relative_percentage_weighted_hit_rate]

    # data["Hit rate change"] = relative_hit_rate
    data["Percentage hit rate change"] = relative_percentage_hit_rate
    # data["Weighted hit rate change"] = relative_weighted_hit_rate
    data["Percentage weighted hit rate change"] = relative_percentage_weighted_hit_rate
    data.to_csv(os.path.join(dirname + filename + "_analyzed_hello.txt"))

    return data.drop(["Hit rate", "Weighted Hit Rate"], axis=1)


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
    data = get_admittor_df_from_file(filename=sim_filename)
    data = data.groupby("Policy").mean()
    print(data)
    for p in exclude_policies:
        data = data[~data.index.str.contains(p)]
    admittors = []
    for idx, row in data.iterrows():
        admittor = ""
        if "_" in idx:
            admittor = idx.split("_")[1]
        else:
            admittor = "None"
        admittors.append(admittor)
    data["Admittor"] = admittors
    data = data.groupby("Admittor").mean()
    #plot_df(data)
    data["Percentage hit rate change"] = data["Percentage hit rate change"].map('{:,.2f}%'.format)
    data["Percentage weighted hit rate change"] = data["Percentage weighted hit rate change"].map('{:,.2f}%'.format)
    print(data)


def plot_df(data):
    sns.set_context('paper')
    sns.set(style='ticks')
    print("PLOT_DF", data.head(1))
    #fg = sns.FacetGrid(data=data, aspect=1.66)
    #fg.map(plt.scatter, 'Percentage hit rate change')
    fg = sns.barplot(data.index, data['Percentage hit rate change'], alpha=0.8)
    fg.set_xlabel("Admittor")
    fg.set_ylabel("Percentage hit rate change")
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # analyze_directory("/results/web")
    analyze_simulation("/results/web/web_0.txt", ["Mru", "Mfu"])
