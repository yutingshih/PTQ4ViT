from importlib import import_module, reload
import os


def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _, _, files = next(os.walk("../configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg
