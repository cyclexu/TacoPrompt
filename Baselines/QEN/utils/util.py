import json
import networkx as nx
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


def custom_cat(x, y):
    import torch
    n = y.size(0)
    x = x.reshape(n, -1)
    c = torch.cat((y.view(-1, 1), x), dim=1)
    return c.flatten()


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
