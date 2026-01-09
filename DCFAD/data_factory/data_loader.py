import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SWaT_train.npy",allow_pickle=True)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SWaT_test.npy",allow_pickle=True)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SWaT_test_label.npy",allow_pickle=True)
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class ASD1SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-1.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))

class ASD2SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-2.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD3SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-3.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD4SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-4.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD5SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-5.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD6SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-6.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD7SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-7.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))

class ASD8SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-8.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))

class ASD9SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-9.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD10SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-10.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:  # 其它模式：整窗口滑动
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD11SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-11.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)

        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class ASD12SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode     = mode
        self.step     = step
        self.win_size = win_size
        self.scaler   = StandardScaler()

        f_name = "omi-12.pkl"
        def to_ndarray(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.values
            else:
                return np.asarray(obj)

        train_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "train", f_name)))
        test_raw    = to_ndarray(pd.read_pickle(os.path.join(data_path, "test", f_name)))
        label_raw   = to_ndarray(pd.read_pickle(os.path.join(data_path, "test_label", f_name))).astype(int)

        self.scaler.fit(train_raw)
        self.train       = self.scaler.transform(train_raw)
        self.test        = self.scaler.transform(test_raw)
        self.test_labels = label_raw

        split_idx = int(len(self.train) * 0.8)
        self.val  = self.train[split_idx:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0]  - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "val":
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == "test":
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test_labels[index:index + self.win_size]))
        else:
            start = (index // self.step) * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[start:start + self.win_size]))
class GECCOSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/GECCO_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/GECCO_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/GECCO_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/Swan_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class WADISegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = np.load(data_path + "/test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD1'):
        dataset = ASD1SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD2'):
        dataset = ASD2SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD3'):
        dataset = ASD3SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD4'):
        dataset = ASD4SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD5'):
        dataset = ASD5SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD6'):
        dataset = ASD6SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD7'):
        dataset = ASD7SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD8'):
        dataset = ASD8SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD9'):
        dataset = ASD9SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD10'):
        dataset = ASD10SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD11'):
        dataset = ASD11SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ASD12'):
        dataset = ASD12SegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'GECCO'):
        dataset = GECCOSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'Swan'):
        dataset = SwanSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
