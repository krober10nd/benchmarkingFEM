import matplotlib.pyplot as plt
import numpy as np

_all__ = ["plot_cfl"]
# x = np.array(
#    [
#        [0.023775, 0.00375, 0.0021, 0.001275],
#        [0.006625, 0.00185, 0.00105, 0.00065],
#        [0.003075, 0.000925, 0.000525, 0.000325],
#    ]
# )

ltype = [
    "r-o",
    "b-o",
    "g-o",
    "m-o",
]


def plot_cfl(x, title=None):
    dxs = np.array([1 / 40, 1 / 80, 1 / 160])
    fig, ax = plt.subplots()
    for deg, result in enumerate(x.T):
        print(result)
        plt.plot(dxs, result, ltype[deg], label="degree " + str(deg + 1))
    plt.xlabel("Mesh size")
    plt.ylabel("Max. stable dt (s)")
    plt.legend()
    ax.set_yscale("log")
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.show()


# plot_cfl(x, title="KMV-tria")
