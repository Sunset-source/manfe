import matplotlib.pyplot as plt

from prelude import *


def draw():
    fig, ax = plt.subplots(1, 1)
    # r = mixture_gaussian([0.5, 1], [-1, 1], [2, 1], 10000)
    r = nakagami.rvs(4, size=10000)
    ax.hist(r, bins=20, density=True, histtype='stepfilled', alpha=0.2)
    plt.show()


if __name__ == "__main__":
    draw()
