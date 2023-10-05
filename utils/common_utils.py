import re
import pickle
import itertools
import functools
import numpy as np
from ray.utils import binary_to_hex, hex_to_binary


class KeyAwareDefaultDict(dict):
    def __init__(self, default_factory, *args, **kwargs):
        self.default_factory = default_factory
        super(KeyAwareDefaultDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        self[key] = val = self.default_factory(key)
        return val


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, item):
        value = dict.__getattribute__(self, item)
        if isinstance(value, dict):
            return DotDict(value)
        else:
            return value


def flatten_dict(dt, delimiter="/"):
    dt = dt.copy()
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def sorted_nicely(iterable):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(iterable, key=alphanum_key)


def take(n, iterable):
    return list(itertools.islice(iterable, n))


def chunked(iterable, n):
    iterator = iter(functools.partial(take, n, iter(iterable)), [])
    return iterator


def numpy_to_string(x):
    assert isinstance(x, np.ndarray)
    return binary_to_hex(pickle.dumps(x))


def string_to_numpy(x):
    assert isinstance(x, str)
    return pickle.loads(hex_to_binary(x))
