import keras
from generator import Generator
from preprocess import preproc_gray, preproc_lab, preproc_morphology
from zipfile import ZipFile
import numpy as np
import gdown

if __name__ == '__main__':
    with open("dataset.txt", "r") as f:
        data_paths = f.readlines()

    data_paths = [x.strip() for x in data_paths]

    id = ''  # google drive id to download the model
    name = ''  # name to save the model
    path = ''  # path to the saved model

    gdown.download(f"https://drive.google.com/uc?id={id}", name)

    with ZipFile(name, 'r') as zip_ref:
        zip_ref.extractall(path)

    model = keras.models.load_model(path)

    pgen = Generator(data_paths, data_paths, 32, 3, is_predicting=True, preprocess=preproc_gray)

    preds = model.predict(pgen)

    print(preds)

    pred = [np.argmax(x) for x in preds]

    with open("predictions.txt", "w") as f:
        for i, p in enumerate(pred):
            f.write(f"{data_paths[i]}, {p}\n")
