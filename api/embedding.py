import io
import os
import re
import sys
import csv
import json
import torch
import logging
import numpy as np
import torch.nn as nn
import tensorflow as tf
from pathlib import Path
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from utils import ImageReader

import cnn_architectures
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, 'cnn_architectures'))

ROOT_PATH = Path(os.path.dirname(__file__))

class Embedding:
    def __init__(self):
        self.object_model = ObjectEmbedding()
        self.location_model = LocationEmbedding()
        self.scene_model = SceneEmbedding()

    def _object_embedding(self, image):
        return self.object_model.embed(image)

    def _location_embedding(self, image):
        return self.location_model.embed(image)

    def _scene_embedding(self, image):
        return self.scene_model.embed(image)

    def _all_embedding(self, obj, loc, scene):
        return np.concatenate((np.concatenate((loc, scene)), obj))

    def embed_pil_image(self, image):
        obj = self._object_embedding(image)
        loc = self._location_embedding(image)
        scene = self._scene_embedding(image)

        return {
            'object': obj,
            'location': loc,
            'scene': scene,
            'all': self._all_embedding(obj, loc, scene)
        }

    def embed_url_image(self, image_url):
        image = ImageReader.read_from_url(image_url)
        return self.embed_pil_image(image)

class ObjectEmbedding:
    def __init__(self):
        self.DEVICE = torch.device('cpu')
        resnet152 = models.resnet152(pretrained=True)
        modules = list(resnet152.children())[:-1]

        for p in resnet152.parameters():
            p.requires_grad = False

        self.resnet = nn.Sequential(*modules).to(self.DEVICE)

        self.scaler = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()

    def embed(self, image):
        result = Variable(self.to_tensor(self.scaler(image)).to(self.DEVICE).unsqueeze(0))
        return self.resnet(result).data.view(result.size(0), 2048).detach().cpu().numpy()[0]


class SceneEmbedding:
    def __init__(self, model_path=f'{ROOT_PATH}/models/scene/resnet50_places365.pth.tar',
                        labels_file=f'{ROOT_PATH}/models/scene/categories_places365.txt',
                        hierarchy_file=f'{ROOT_PATH}/models/scene/scene_hierarchy_places365.csv',
                        arch='resnet50'):
        if model_path is not None:
            model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval().to(self._device)
            self.model = model

            # method for centre crop
            self._centre_crop = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            logging.warning('No model built.')

        # load hierarchy
        if hierarchy_file is not None and os.path.isfile(hierarchy_file):
            self._load_hierarchy(hierarchy_file)
        else:
            logging.warning('Hierarchy file not specified.')

        # load the class label
        if labels_file is not None and os.path.isfile(labels_file):
            classes = list()
            with open(labels_file, 'r') as class_file:
                for line in class_file:
                    cls_name = line.strip().split(' ')[0][3:]
                    cls_name = cls_name.split('/')[0]
                    classes.append(cls_name)
            self.classes = tuple(classes)
        else:
            logging.warning('Labels file not specified.')

    def embed(self, image):
        try:
            input_img = Variable(self._centre_crop(image).unsqueeze(0)).to(self._device)

            # forward pass for feature extraction
            x = input_img
            i = 0
            for module in self.model._modules.values():
                if i == 9:
                    break
                x = module(x)
                i += 1

            return x.detach().cpu().numpy().squeeze()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            logging.error(f'Cannot create embedding for image')
            return []

    def _load_hierarchy(self, hierarchy_file):
        hierarchy_places3 = []
        hierarchy_places16 = []

        with open(hierarchy_file, 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content)  # skip explanation line
            next(content)  # skip explanation line
            for line in content:
                hierarchy_places3.append(line[1:4])
                hierarchy_places16.append(line[4:])

        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=np.float)
        hierarchy_places16 = np.asarray(hierarchy_places16, dtype=np.float)

        # NORM: if places label belongs to multiple labels of a lower level --> normalization
        self._hierarchy_places3 = hierarchy_places3 / np.expand_dims(np.sum(hierarchy_places3, axis=1), axis=-1)
        self._hierarchy_places16 = hierarchy_places16 / np.expand_dims(np.sum(hierarchy_places16, axis=1), axis=-1)


class LocationEmbedding:
    def __init__(self, model_path=f'{ROOT_PATH}/models/location/base_M/', cnn_input_size=224, use_cpu=True):
        logging.info(f'Initialize {os.path.basename(model_path)} geolocation model.')
        self._cnn_input_size = cnn_input_size
        self._image_path_placeholder = tf.placeholder(tf.uint8, shape=[None, None, None])
        self._image_crops, _ = self._img_preprocessing(self._image_path_placeholder)

        # load model config
        with open(os.path.join(model_path, 'cfg.json'), 'r') as cfg_file:
            cfg = json.load(cfg_file)

        # build cnn
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        model_file = os.path.join(model_path, 'model.ckpt')
        logging.info('\tRestore model from: {}'.format(model_file))

        with tf.variable_scope(os.path.basename(model_path)) as scope:
            self._scope = scope

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.variable_scope(self._scope):
            with tf.device(device):
                self._net, _ = cnn_architectures.create_model(cfg['architecture'],
                                                              self._image_crops,
                                                              is_training=False,
                                                              num_classes=None,
                                                              reuse=None)

        var_list = {
            re.sub('^' + self._scope.name + '/', '', x.name)[:-2]: x for x in tf.global_variables(self._scope.name)
        }

        # restore weights
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self._sess, str(model_file))

    def embed(self, image):
        # feed forward image in cnn and extract result
        # use the mean for the three crops
        try:
            embedding = self._sess.run([self._net], feed_dict={self._image_path_placeholder: image})
            return embedding[0].squeeze().mean(axis=0)
        except KeyboardInterrupt:
            raise
        except:
            logging.error(f'Cannot create embedding for image.')
            return []

    def _img_preprocessing(self, image):
        img = tf.image.convert_image_dtype(image, dtype=tf.float32)
        img.set_shape([None, None, 3])

        # normalize image to -1 .. 1
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        # get multicrops depending on the image orientation
        height = tf.to_float(tf.shape(img)[0])
        width = tf.to_float(tf.shape(img)[1])

        # get minimum and maximum coordinate
        max_side_len = tf.maximum(width, height)
        min_side_len = tf.minimum(width, height)
        is_w, is_h = tf.cond(tf.less(width, height), lambda: (0, 1), lambda: (1, 0))

        # resize image
        ratio = self._cnn_input_size / min_side_len
        offset = (tf.to_int32(max_side_len * ratio + 0.5) - self._cnn_input_size) // 2
        img = tf.image.resize_images(img, size=[tf.to_int32(height * ratio + 0.5), tf.to_int32(width * ratio + 0.5)])

        # get crops according to image orientation
        img_array = []
        bboxes = []

        for i in range(3):
            bbox = [
                i * is_h * offset, i * is_w * offset,
                tf.constant(self._cnn_input_size),
                tf.constant(self._cnn_input_size)
            ]

            img_crop = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
            img_crop = tf.expand_dims(img_crop, 0)

            img_array.append(img_crop)
            bboxes.append(bbox)

        return tf.concat(img_array, axis=0), bboxes
