from typing import List

from pathlib import Path


def FindFiles(dir_path, formats=["*.xml"], recursive=False):
    dir_path = Path(dir_path)
    filepath = {}
    for format in formats:
        filepath.update({path.name: path for path in dir_path.rglob(format)} if recursive else {path.name: path for path in dir_path.glob(format)})
    return filepath

def CreateDir(cache_dir:str):
    dir = Path(cache_dir)
    dir.mkdir(parents=True, exist_ok=True)
    return dir

