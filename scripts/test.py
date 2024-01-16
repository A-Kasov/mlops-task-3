import os
import gdown

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def download_dataset(id):
    gdown.download(id=id, output=os.path.join(BASE_DIR, "datasets/train.csv"))

download_dataset('14YPaOI3VspacNH226zhHsAegaoqn9nui')