import numpy as np
from os import listdir
from os.path import join


def load(dir='bottleneck2/'):
    x = []
    y = []
    files = listdir(dir)
    files = sorted(files)
    count = len(files)
    for i, file in enumerate(files):
        if file.endswith('.txt'):
            continue
        print(str(i) + "/" + str(count))
        is_x = file.startswith('x')
        path = join(dir, file)
        arr = np.load(path)
        if is_x:
            x.append(arr)
        else:
            y.append(arr)

    return np.concatenate(x), np.concatenate(y)

