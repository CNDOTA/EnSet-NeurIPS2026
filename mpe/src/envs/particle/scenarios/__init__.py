try:
    import imp
    def load_source(name, path):
        return imp.load_source(name, path)
except ImportError:  # python 3.12+
    import importlib.util
    def load_source(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return load_source('', pathname)
