import numpy as np
from os import listdir
from os.path import join


def load(dir='bottleneck/'):
    x = []
    y = []
    files = listdir(dir)
    files = sorted(files)
    for file in files:
        is_x = file.startswith('x')
        path = join(dir, file)
        arr = np.load(path)
        if is_x:
            x.append(arr)
        else:
            y.append(arr)

    return np.concatenate(x), np.concatenate(y)
