import os
import requests
import zipfile
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def download_fer2013():
    """Download FER2013 dataset from Kaggle"""
    os.system(
        "curl -L -o ./data/fer2013.zip https://www.kaggle.com/api/v1/datasets/download/deadskull7/fer2013"
    )
    with zipfile.ZipFile("./data/fer2013.zip", "r") as zip_ref:
        zip_ref.extractall("./data")
    os.remove("./data/fer2013.zip")


def download_fer_plus():
    """Download FER+ labels from Microsoft"""
    url = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
    response = requests.get(url)
    with open("./data/fer2013new.csv", "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    download_fer2013()
    download_fer_plus()
    print("Datasets downloaded.")
