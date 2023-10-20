import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from collections import OrderedDict
# import datetime
from datetime import date, time, datetime
import json


class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        self.cfg_fname = Path(args.config)

        # load config file and apply custom cli options
        config = _read_json(self.cfg_fname)
        self.__config = _update_config(config, options, args)
        date_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        save_dir = Path(self.config['data_path'] + date_time)
        # run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        # model_save_path = Path(self.config['data_path'] + run_id)
        self.__save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # save updated config file to the checkpoint dir
        _write_json(self.config, self.save_dir / 'config_method.json')




    def __getitem__(self, name):
        return self.config[name]


    # setting read-only attributes
    @property
    def config(self):
        return self.__config

    @property
    def save_dir(self):
        return self.__save_dir

    @property
    def log_dir(self):
        return self.__log_dir

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


def _read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def _write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)