import cv2
import os
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_classes, labels=None, is_predicting=False, w_max=400, h_max=400,
                 preprocess=None):
        self.x, self.y = x_set, y_set
        self.is_predicting = is_predicting
        self.batch_size = batch_size
        self.w_max = w_max
        self.h_max = h_max
        self.labels = labels
        self.num_classes = num_classes
        self.preprocess = preprocess
        if not self.is_predicting:
            self.classes = self.get_classes()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [self.read_x(filepath) for filepath in batch_x]
        if not self.is_predicting:
            y = [self.read_y(filepath) for filepath in batch_y]
            return np.array(x), np.array(y)
        else:
            return np.array(x)

    def read_x(self, filepath):
        image = cv2.imread(filepath)
        h, w, ch = image.shape
        delta_w = self.w_max - w
        delta_h = self.h_max - h

        if delta_w < 0 or delta_h < 0:
            image = cv2.resize(image, (self.w_max, self.h_max))
        else:
            image = cv2.copyMakeBorder(image, delta_h // 2, abs(delta_h - delta_h // 2), delta_w // 2,
                                       abs(delta_w - delta_w // 2), cv2.BORDER_CONSTANT)

        if self.preprocess:
            image = self.preprocess(image)

        return image

    def read_y(self, filepath):
        paths = str.split(filepath, os.path.sep)
        filename = paths[-1]
        y = int(self.labels[filename])
        y = to_categorical(y, num_classes=3)
        return y

    def get_classes(self):
        l = []
        for f in self.x:
            paths = str.split(f, os.path.sep)
            filename = paths[-1]
            y = int(self.labels[filename])
            l.append(y)
        return l
