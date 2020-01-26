import numpy as np
from matplotlib import pyplot as plt


def harmonic_mean(first, second):
    return (2.0 * first * second) / (first + second)


def metric(first, second):
    return harmonic_mean(first + 0.3, second + 0.3)


def main():
    size = 1000
    # min_x, max_x, min_y, max_y
    x_vals = np.random.uniform(size=(2, size)) + np.random.uniform(
        -10, 10, size=(1, size))
    y_vals = np.random.uniform(size=(2, size)) + np.random.uniform(
        -10, 10, size=(1, size))
    x_min = x_vals.min(0)
    x_max = x_vals.max(0)
    y_min = y_vals.min(0)
    y_max = y_vals.max(0)
    surface_area = (x_max - x_min) * (y_max - y_min)
    min_sorted_idx = np.flip(np.argsort(x_min))
    max_sorted_idx = np.argsort(x_max)
    x_min_sorted = x_min[min_sorted_idx]
    x_max_sorted = x_max[max_sorted_idx]
    min_summed_surface_area = np.cumsum(surface_area[min_sorted_idx])
    max_summed_surface_area = np.cumsum(surface_area[max_sorted_idx])

    scores = np.empty(size)

    last_max_index = size - 1
    for i in range(size):
        bound = x_min_sorted[i]
        j = 0
        for j in reversed(range(1, last_max_index + 2)):
            if x_max_sorted[j - 1] <= bound:
                break
        last_max_index = j - 1
        scores[i] = metric(
            min_summed_surface_area[i], 0 if last_max_index == 0 else
            max_summed_surface_area[last_max_index])

    plt.plot(scores)
    plt.show()


if __name__ == "__main__":
    main()
