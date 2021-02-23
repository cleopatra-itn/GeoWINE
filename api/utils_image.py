from collections import OrderedDict
from PIL.Image import open as open_image
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import csv
import json
import logging
import numpy as np
import os
import re
import sys
import tensorflow as tf
import cnn_architectures
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, 'cnn_architectures'))


class SceneClassificator:

    def __init__(self, model_path=None, labels_file=None, hierarchy_file=None, arch='resnet50'):
        if model_path is not None:
            model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval().to(self._device)
            self.model = model

            # method for centre crop
            self._centre_crop = trn.Compose([
                trn.Resize((256, 256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    # def get_scene_word_embeddings(self, fasttext_bin_folder, token_types=None, language='en'):
    #     if self.classes is None:
    #         logging.error('Cannot create word embedinngs. Please specify labels file in class constructor')
    #
    #     we = word_embedder(fasttext_bin_folder, token_types=token_types, language=language)
    #     scene_word_embeddings = []
    #
    #     for cls in self.classes:
    #         cls_emb = we.generate_embeddings(cls)
    #         if cls_emb.shape[0] != 1:
    #             logging.error(f'Invalid scene shape {cls_emb.shape} for scene {cls}')
    #             exit()
    #         scene_word_embeddings.append(cls_emb)
    #
    #     return np.concatenate(scene_word_embeddings, axis=0)

    def get_img_embedding(self, img_path):
        try:
            img = open_image(img_path).convert('RGB')
            input_img = V(self._centre_crop(img).unsqueeze(0)).to(self._device)

            # forward pass for feature extraction
            x = input_img
            i = 0
            for module in self.model._modules.values():
                if i == 9:
                    break
                x = module(x)
                i += 1

            return [x.detach().cpu().numpy().squeeze()]  # return as list for compatability to face verification
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            logging.error(f'Cannot create embedding for {img_path}')
            return []

    def get_img_classification(self, img_path):
        img = open_image(img_path).convert('RGB')
        input_img = V(self._centre_crop(img).unsqueeze(0))

        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        out = OrderedDict()
        for i in range(0, 5):
            label = self.classes[idx[i]]
            out[label] = np.round(probs[i].detach().item(), 3)
        return out

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

    def get_logits(self, img_path):
        try:
            img = open_image(img_path).convert('RGB')
            input_img = V(self._centre_crop(img).unsqueeze(0)).to(self._device)

            logit = self.model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            return h_x.detach().cpu().numpy().squeeze()
        except KeyboardInterrupt:
            raise
        except:
            logging.error(f'Cannot create logits for {img_path}')
            return []

    def get_hierarical_prediction(self, logits, hierarchy_level='places3'):
        if hierarchy_level == 'places365':
            return np.argmax(logits, axis=0)
        elif hierarchy_level == 'places16':
            places16_h = np.matmul(logits, self._hierarchy_places16)
            return np.argmax(places16_h, axis=0)
        elif hierarchy_level == 'places3':
            places3_h = np.matmul(logits, self._hierarchy_places3)
            return np.argmax(places3_h, axis=0)
        else:
            logging.error('Unknown hierarchy level. Exiting ...')
            return None


class GeoEstimator():

    def __init__(self, model_path, cnn_input_size=224, use_cpu=False):
        logging.info(f'Initialize {os.path.basename(model_path)} geolocation model.')
        self._cnn_input_size = cnn_input_size
        self._image_path_placeholder = tf.placeholder(tf.string, shape=None)
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

    def get_img_embedding(self, image_path):
        # feed forward image in cnn and extract result
        # use the mean for the three crops
        try:
            embedding = self._sess.run([self._net], feed_dict={self._image_path_placeholder: image_path})
            return [embedding[0].squeeze().mean(axis=0)]  # needs to be a list for compatibility to face verification
        except KeyboardInterrupt:
            raise
        except:
            logging.error(f'Cannot create embedding for {image_path}')
            return []

    def _img_preprocessing(self, img_path):
        # read image
        img = tf.io.read_file(img_path)

        # decode image
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
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
