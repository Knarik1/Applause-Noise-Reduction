import os
import numpy as np
import urllib.request
from PIL import Image

# base URL for downloading the data-files from the internet.
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download(filename, download_dir):
    # path for local file.
    save_path = os.path.join(download_dir, filename)
    # check if the file already exists, otherwise we need to download it now.
    if not os.path.exists(save_path):
        # check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        print("Downloading", filename, "...")

        # download the file from the internet.
        url = base_url + filename
        file_path, _ = urllib.request.urlretrieve(url=url, filename=save_path)

        print(" Done!")


def one_hot_encoded(labels, num_classes):
    return np.eye(num_classes, dtype=float)[labels]


def save_in_folders(path, images, labels):
    for i in range(0, len(images)):
        image = images[i]
        label = labels[i]
        save_class(label, path, image, i)


def save_class(j, save_dir, image, i):
    class_dir = os.path.join(save_dir + str(j))
    file_name = str(class_dir + '/' + str(i) + '.png')

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # reshaping image from (28,28,1) to (28,28)
    image = image.reshape(image.shape[0], image.shape[1])
    im = Image.fromarray(image)
    im.save(file_name, "PNG")