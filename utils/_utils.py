import os
import json

def read_text(path):
    f = open(path, "r")
    text = f.read()
    return text

def read_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def read_text_lines(path):
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    return lines

def get_list_of_files(path):
    list_files = os.listdir(path)
    if '.DS_Store' in list_files:
        list_files.remove('.DS_Store')
    return list_files

class TextManager:
    def __init__(self, path_json="./config/pathes.json"):
        self.dict_path = read_json(path_json)

    def __call__(self, key, return_path=False):
        path = self.dict_path[key]
        if return_path:
            return path
        else:
            text = read_text(path)
            return text
        
    def read_data(self, path):
        data = read_text_lines(path)
        return data
