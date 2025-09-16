import shutil
import os
import zipfile
import string
import requests

dataset_url = "https://www.kaggle.com/api/v1/datasets/download/sujaykapadnis/smoking"
data_path = "../data/"
raw_data_path = "../data/raw/"
file_name = "archive.zip"

# Initial setup
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(raw_data_path):
    os.mkdir(raw_data_path)

# Download dataset
if not os.path.exists(raw_data_path + file_name):
    print("Dataset not found, downloading...")
    req = requests.get(dataset_url, allow_redirects=True)
    open(raw_data_path + file_name, "wb+").write(req.content)
    print("Download completed!")

zip_file = zipfile.ZipFile(raw_data_path + file_name, 'r')

for set in ["train", "test", "val"]:
    if os.path.isdir(data_path + set):
        break

    os.mkdir(data_path + set)
    for cls in ["smoking", "no_smoking"]:
        os.mkdir(data_path + set + "/" + cls)


for file_info in zip_file.infolist():
    file_name = file_info.filename.split("/")

    folder = ""
    cls = ""

    if file_name[0] == "Training":
        folder = "train/"
    elif file_name[0] == "Testing":
        folder = "test/"
    elif file_name[0] == "Validation":
        folder = "val/"

    if file_name[-1].startswith("smoking"):
        cls = "smoking/"
    else:
        cls = "no_smoking/"

    with open(data_path+folder+cls+file_name[-1], 'wb') as f:
        f.write(zip_file.read(file_info.filename))
