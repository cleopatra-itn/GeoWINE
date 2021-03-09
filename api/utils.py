import io
import torch
import requests
import torchvision
import numpy as np
import pandas as pd
import s2sphere as s2
from PIL import Image
import torch.nn as nn
from pathlib import Path
from scipy import spatial
from resizeimage import resizeimage
from collections import OrderedDict

class ImageReader:
    @staticmethod
    def read_and_resize(image):
        img = Image.open(image, 'r')
        img = resizeimage.resize_contain(img, [256, 256])
        return img.convert('RGB')

    @staticmethod
    def read(image):
        return Image.open(image).convert("RGB")

    @staticmethod
    def read_from_url(image_url):
        response = requests.get(image_url, stream=True)
        image_bytes = io.BytesIO(response.content)

        return ImageReader.read_and_resize(image_bytes)

    @staticmethod
    def read_from_path(image_path):
        return ImageReader.read_and_resize(image_path)

class ImageCropper:
    def __init__(self):
        self.tfm = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )

    def crop(self, image):
        image = torchvision.transforms.Resize(256)(image)
        crops = torchvision.transforms.FiveCrop(224)(image)

        return torch.stack([self.tfm(crop) for crop in crops], dim=0)

class Cosine:
    @staticmethod
    def distance(u, v):
        return spatial.distance.cosine(u, v)

    @staticmethod
    def similarity(u, v):
        return 1 - Cosine.distance(u, v)

class Partitioning:
    def __init__(self, csv_file, shortname=None, skiprows=None,
                        index_col="class_label", col_class_label="hex_id",
                        col_latitute="latitude_mean", col_longitude="longitude_mean"):
        """
        Required information in CSV:
            - class_indexes from 0 to n
            - respective class labels i.e. hexid
            - latitude and longitude
        """

        print(f"Loading partitioning from file: {csv_file}")
        self._df = pd.read_csv(csv_file, index_col=index_col, skiprows=skiprows)
        self._df = self._df.sort_index()

        self._nclasses = len(self._df.index)
        self._col_class_label = col_class_label
        self._col_latitude = col_latitute
        self._col_longitude = col_longitude

        # map class label (hexid) to index
        self._label2index = dict(
            zip(self._df[self._col_class_label].tolist(), list(self._df.index))
        )

        self.name = csv_file.stem  # filename without extension
        if shortname:
            self.shortname = shortname
        else:
            self.shortname = self.name

    def __len__(self):
        return self._nclasses

    def __repr__(self):
        return f"{self.name} short: {self.shortname} n: {self._nclasses}"

    def get_class_label(self, idx):
        return self._df.iloc[idx][self._col_class_label]

    def get_lat_lng(self, idx):
        x = self._df.iloc[idx]
        return float(x[self._col_latitude]), float(x[self._col_longitude])

    def contains(self, class_label):
        if class_label in self._label2index:
            return True
        return False

    def label2index(self, class_label):
        try:
            return self._label2index[class_label]
        except KeyError as e:
            raise KeyError(f"unkown label {class_label} in {self}")


class Hierarchy:
    def __init__(self, partitionings):

        """
        Provide a matrix of class indices where each class of the finest partitioning will be assigned
        to the next coarser scales.

        Resulting index matrix M has shape: max(classes) * |partitionings| and is ordered from coarse to fine
        """
        self.partitionings = partitionings
        self.M = self.__build_hierarchy()

    def __build_hierarchy(self):
        def _hextobin(hexval):
            thelen = len(hexval) * 4
            binval = bin(int(hexval, 16))[2:]
            while (len(binval)) < thelen:
                binval = "0" + binval

            binval = binval.rstrip("0")
            return binval

        def _create_cell(lat, lng, level):
            p1 = s2.LatLng.from_degrees(lat, lng)
            cell = s2.Cell.from_lat_lng(p1)
            cell_parent = cell.id().parent(level)
            hexid = cell_parent.to_token()
            return hexid

        cell_hierarchy = []

        finest_partitioning = self.partitionings[-1]
        print("Create hierarchy from partitionings...")
        if len(self.partitionings) > 1:
            # loop through finest partitioning
            for c in range(len(finest_partitioning)):
                cell_bin = _hextobin(self.partitionings[-1].get_class_label(c))
                level = int(len(cell_bin[3:-1]) / 2)
                parents = []

                # get parent cells
                for l in reversed(range(2, level + 1)):
                    lat, lng = finest_partitioning.get_lat_lng(c)
                    hexid_parent = _create_cell(lat, lng, l)
                    # to coarsest partitioning
                    for p in reversed(range(len(self.partitionings))):
                        if self.partitionings[p].contains(hexid_parent):
                            parents.append(
                                self.partitionings[p].label2index(hexid_parent)
                            )

                    if len(parents) == len(self.partitionings):
                        break

                cell_hierarchy.append(parents[::-1])
        print("Finished.")
        M = np.array(cell_hierarchy, dtype=np.int32)

        assert max([len(p) for p in self.partitionings]) == M.shape[0]
        assert len(self.partitionings) == M.shape[1]

        return M

def build_base_model(arch: str):
    model = torchvision.models.__dict__[arch](pretrained=True)

    # get input dimension before classification layer
    if arch in ["mobilenet_v2"]:
        nfeatures = model.classifier[-1].in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif arch in ["densenet121", "densenet161", "densenet169"]:
        nfeatures = model.classifier.in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif "resne" in arch:
        # usually all ResNet variants
        nfeatures = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
    else:
        raise NotImplementedError

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.flatten = nn.Flatten(start_dim=1)
    return model, nfeatures


def load_weights_if_available(model, classifier, weights_path):
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_classifier = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("classifier"):
            state_dict_classifier[k.replace("classifier.", "")] = w
        else:
            print(f"Unexpected prefix in state_dict: {k}")
    model.load_state_dict(state_dict_features, strict=True)
    return model, classifier