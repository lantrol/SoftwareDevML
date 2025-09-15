import shutil
import os
import zipfile
import string

data_path = "../data/"
file_name = "archive.zip"

zip_file = zipfile.ZipFile(data_path + file_name, 'r')

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
