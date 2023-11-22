import json


class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__

