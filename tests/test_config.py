"""Tests for config module — dict/namespace conversion and merging."""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDictToNamespace:
    def test_simple(self):
        from config import dict_to_namespace
        d = {'a': 1, 'b': 'hello'}
        ns = dict_to_namespace(d)
        assert ns.a == 1
        assert ns.b == 'hello'

    def test_nested(self):
        from config import dict_to_namespace
        d = {'model': {'hidden_size': 64, 'type': 'unet'}, 'lr': 1e-4}
        ns = dict_to_namespace(d)
        assert ns.model.hidden_size == 64
        assert ns.model.type == 'unet'
        assert ns.lr == 1e-4

    def test_hyphen_to_underscore(self):
        from config import dict_to_namespace
        d = {'my-key': 42}
        ns = dict_to_namespace(d)
        assert ns.my_key == 42

    def test_list_handling(self):
        from config import dict_to_namespace
        d = {'items': [{'name': 'a'}, {'name': 'b'}]}
        ns = dict_to_namespace(d)
        assert ns.items[0].name == 'a'
        assert ns.items[1].name == 'b'


class TestConfigToDict:
    def test_roundtrip(self):
        from config import dict_to_namespace, config_to_dict
        original = {'model': {'hidden_size': 64}, 'data': {'resolution': 512}}
        ns = dict_to_namespace(original)
        result = config_to_dict(ns)
        assert result == original

    def test_simple_namespace(self):
        from config import config_to_dict
        ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2))
        d = config_to_dict(ns)
        assert d == {'a': 1, 'b': {'c': 2}}


class TestMergeConfigs:
    def test_simple_override(self):
        from config import merge_configs
        base = {'a': 1, 'b': 2}
        user = {'b': 99}
        result = merge_configs(base, user)
        assert result == {'a': 1, 'b': 99}

    def test_deep_merge(self):
        from config import merge_configs
        base = {'model': {'size': 64, 'type': 'unet'}, 'lr': 1e-4}
        user = {'model': {'size': 128}}
        result = merge_configs(base, user)
        assert result['model']['size'] == 128
        assert result['model']['type'] == 'unet'  # preserved from base
        assert result['lr'] == 1e-4

    def test_add_new_key(self):
        from config import merge_configs
        base = {'a': 1}
        user = {'b': 2}
        result = merge_configs(base, user)
        assert result == {'a': 1, 'b': 2}

    def test_does_not_mutate_base(self):
        from config import merge_configs
        base = {'a': {'x': 1}}
        user = {'a': {'x': 99}}
        merge_configs(base, user)
        assert base['a']['x'] == 1  # unchanged
