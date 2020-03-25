import pandas as pd
import numpy as np
import os

dirname = os.path.dirname(__file__)


def _msr_web():
    d1 = pd.read_csv(os.path.join(dirname + "/msr/web_0.csv/CAMRESWEBA03-lvm0.csv"), header=None, usecols=[4]).iloc[:,
         0]

    d2 = pd.read_csv(os.path.join(dirname + "/msr/web_2.csv/CAMRESWEBA03-lvm2.csv"), header=None, usecols=[4]).iloc[:,
         0]

    d1keys = set(d1.unique())

    d2keys = set(d2.unique())

    intersect = d1keys.intersection(d2keys)
    print(len(intersect))
    print(intersect)


def _lirs():
    d1 = set()
    with open(os.path.join(dirname + "/lirs/cs.trace/cs.trc")) as f:
        for line in f:
            if '*' not in line:
                d1.add(int(line))
    print("D1 len: ", len(d1))

    d2 = set()
    with open(os.path.join(dirname + "/lirs/multi1.trace/multi1.trc")) as g:
        for line in g:
            if '*' not in line:
                d2.add(int(line))
    print("D2 len: ", len(d2))

    d3 = set()
    with open(os.path.join(dirname + "/lirs/multi2.trace/multi2.trc")) as g:
        for line in g:
            if '*' not in line:
                d3.add(int(line.strip()))
    print("D3 len: ", len(d3))

    insc = d1.intersection(d2).intersection(d3)

    print(insc)
    print(len(insc))


if __name__ == '__main__':
    _lirs()
