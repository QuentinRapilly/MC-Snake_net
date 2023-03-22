from contextlib import contextmanager
from time import time


@contextmanager
def time_manager(time_dict, feature_name):
    start = time()
    yield
    end = time()
    elapsed_time = end - start
    stored = time_dict.get(feature_name)
    time_dict[feature_name] = elapsed_time if stored == None else stored + elapsed_time


def print_time_dict(time_dict):
    for key in time_dict:
        print("Step '{}', time elapsed {}s".format(key,round(time_dict[key],3)))