import json
from typing import List

from llama_index.indices.base import BaseIndex


class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__


def find_typed(objs: List, typ):
    for obj in objs:
        if isinstance(obj, typ):
            return obj
    raise Exception(f"Can't found type={typ.__name__}")
