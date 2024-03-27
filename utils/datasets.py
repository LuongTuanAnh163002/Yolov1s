from tensorflow.keras.utils import Sequence
import math
import cv2 as cv
import numpy as np
import os
import cv2
import glob
from pathlib import Path

class SequenceData(Sequence):
    def __init__(self, model, dir, target_size, batch_size, shuffle=True):
        self.model = model
        self.datasets = []
        if self.model == 'train':
            with open(os.path.join(dir, '2007_train.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        elif self.model == 'val':
            with open(os.path.join(dir, '2007_val.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.shuffle = shuffle

    def __len__(self):
        num_imgs = len(self.datasets)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.datasets[k] for k in batch_indexs]
        X, y = self.data_generation(batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        label = dataset[1:]

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_h, image_w = image.shape[0:2]
        image = cv.resize(image, self.image_size)
        image = image / 255.

        label_matrix = np.zeros([7, 7, 25])
        for l in label:
            l = l.split(',')
            l = np.array(l, dtype=np.int64)
            xmin = l[0]
            ymin = l[1]
            xmax = l[2]
            ymax = l[3]
            cls = l[4]
            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h
            loc = [7 * x, 7 * y]
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j

            if label_matrix[loc_i, loc_j, 24] == 0:
                label_matrix[loc_i, loc_j, cls] = 1
                label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                label_matrix[loc_i, loc_j, 24] = 1  # response

        return image, label_matrix

    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:
            image, label = self.read(dataset)
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
class LoadImages:  # for inference
    def __init__(self, path, img_size=448):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            img = img0.copy()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
                    img = img0.copy()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            img = cv2.imread(path)
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Convert


        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
