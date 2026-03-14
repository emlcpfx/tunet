import copy
import logging
from types import SimpleNamespace


def dict_to_namespace(d):
    """Convert a nested dict to nested SimpleNamespace."""
    if isinstance(d, dict):
        safe_d = {}
        for k, v in d.items():
            safe_key = str(k).replace('-', '_')
            if not safe_key.isidentifier():
                logging.warning(f"Config key '{k}' -> '{safe_key}' might not be a valid identifier.")
            safe_d[safe_key] = dict_to_namespace(v)
        return SimpleNamespace(**safe_d)
    elif isinstance(d, (list, tuple)):
        return type(d)(dict_to_namespace(item) for item in d)
    return d


def config_to_dict(sns):
    """Convert a nested SimpleNamespace to a plain dict."""
    if isinstance(sns, SimpleNamespace):
        return {k: config_to_dict(v) for k, v in sns.__dict__.items()}
    elif isinstance(sns, (list, tuple)):
        return type(sns)(config_to_dict(item) for item in sns)
    return sns


def dict_to_sns(d):
    """Alias for dict_to_namespace (used by checkpoint loading)."""
    return dict_to_namespace(d)


def merge_configs(base, user):
    """Deep-merge user config dict into base config dict."""
    merged = copy.deepcopy(base)
    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
