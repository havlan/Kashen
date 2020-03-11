import numpy as np
import matplotlib.pyplot as plt
import random


def create_loopy_data(random_range, n, loopy_start, loopy_end):
    data = []
    for i in range(0, n):
        if i < loopy_start or i > loopy_end:
            data.append(random.uniform(0, random_range))
        else:
            data.append(i)
    return data


def to_file(data):
    with open('loopy.txt', 'w') as f:
        for i in data:
            f.write('%d\n' % i)


if __name__ == '__main__':
    data = create_loopy_data(random_range=5000, n=50000, loopy_start=15000, loopy_end=23000)
    to_file(data)
