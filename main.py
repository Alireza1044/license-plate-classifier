import keras
from generator import Generator
from preprocess import preproc_gray, preproc_lab, preproc_morphology
from zipfile import ZipFile
import numpy as np
import gdown
import os

BATCH_SIZE = 32
N_CLASSES = 3


def download_model(id, name, path):
    gdown.download(f"https://drive.google.com/uc?id={id}", name)

    with ZipFile(name, 'r') as zip_ref:
        zip_ref.extractall(path)


if __name__ == '__main__':
    with open("dataset.txt", "r") as f:
        data_paths = f.readlines()

    data_paths = [x.strip() for x in data_paths]

    pgen_gray = Generator(data_paths, data_paths, BATCH_SIZE, N_CLASSES, is_predicting=True, preprocess=preproc_gray)
    pgen_lab = Generator(data_paths, data_paths, BATCH_SIZE, N_CLASSES, is_predicting=True, preprocess=preproc_lab)

    # https://drive.google.com/file/d/1-Q-HQ-tTU3A4jvgAuQD-KEP7Uu-3-cKs/view?usp=sharing # ResNet50_RGB_LAB_SPLIT.zip
    # https://drive.google.com/file/d/1-H8dsQZyyW3mfiVytR-x68CotuxEFaWB/view?usp=sharing # ResNet50_RGB_LAB_WHOLE.zip
    # https://drive.google.com/file/d/1FuQE7PfOcOK54fHm37-kYsIEetaqnZTN/view?usp=sharing # STN_CONV_GRAY_SPLIT.zip

    id = 'h14OY_RCZfult7A_bxSQVCvqh3071OFURa'  # google drive id to download the model
    name = 'RESNET_STN'  # name to save the model
    path = ''  # path to the saved model

    download_model(id, name, path)
    model = keras.models.load_model(path)

    preds = model.predict(pgen_gray)

    print(preds)

    pred = [np.argmax(x) for x in preds]

    with open("predictions.txt", "w") as f:
        for i, p in enumerate(pred):
            f.write(f"{data_paths[i]}, {p}\n")
