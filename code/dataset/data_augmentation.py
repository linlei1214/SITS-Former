import numpy as np
import random


def transform(ts):
    # Flipping
    if random.random() < 0.5:
        ts = np.flip(ts, 2)
    if random.random() < 0.5:
        ts = np.flip(ts, 3)

    # Rotation
    prob = random.random()
    if prob < 0.25:
        ts = np.rot90(ts, k=1, axes=(2, 3))
    elif prob < 0.5:
        ts = np.rot90(ts, k=2, axes=(2, 3))
    elif prob < 0.75:
        ts = np.rot90(ts, k=-1, axes=(2, 3))

    return ts