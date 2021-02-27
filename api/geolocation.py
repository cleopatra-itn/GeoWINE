import os
import torch
import logging
import torchvision
import torch.nn as nn
from pathlib import Path
import pytorch_lightning as pl
from utils import Partitioning, Hierarchy, build_base_model, load_weights_if_available

ROOT_PATH = Path(os.path.dirname(__file__))

class GeolocationEstimator():
    def __init__(self):
        self.geo_model = MultiPartitioningClassifier.load_from_checkpoint(
            checkpoint_path=f'{ROOT_PATH}/models/base_M/epoch=014-val_loss=18.4833.ckpt',
            hparams_file=f'{ROOT_PATH}/models/base_M/hparams.yaml',
            map_location=None,
        )
        self.geo_model.eval()

    def predict(self, image):
        return self.geo_model.inference(image)

class MultiPartitioningClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.partitionings, self.hierarchy = self.__init_partitionings()
        self.model, self.classifier = self.__build_model()

    def __init_partitionings(self):

        partitionings = []
        for shortname, path in zip(
            self.hparams.partitionings["shortnames"],
            self.hparams.partitionings["files"],
        ):
            partitionings.append(Partitioning(Path(f'{ROOT_PATH}/{path}'), shortname, skiprows=2))

        if len(self.hparams.partitionings["files"]) == 1:
            return partitionings, None

        return partitionings, Hierarchy(partitionings)

    def __build_model(self):
        logging.info("Build model")
        model, nfeatures = build_base_model(self.hparams.arch)

        classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(nfeatures, len(self.partitionings[i]))
                for i in range(len(self.partitionings))
            ]
        )

        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, classifier = load_weights_if_available(
                model, classifier, self.hparams.weights
            )

        return model, classifier

    def forward(self, x):
        fv = self.model(x)
        yhats = [self.classifier[i](fv) for i in range(len(self.partitionings))]
        return yhats

    def _inference_step(self, image):
        ncrops = image.shape[0]

        # forward pass
        yhats = [torch.nn.functional.softmax(yhat, dim=1) for yhat in self(image)]

        # respape back to access individual crops
        yhats = [torch.reshape(yhat, (1, ncrops, *list(yhat.shape[1:]))) for yhat in yhats]

        # calculate max over crops
        yhats = [torch.max(yhat, dim=1)[0] for yhat in yhats]

        hierarchy_logits = torch.stack([yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)], dim=-1)

        return torch.prod(hierarchy_logits, dim=-1)

    def inference(self, image):
        hierarchy_preds = self._inference_step(image)
        pred_class = torch.argmax(hierarchy_preds, dim=1).tolist()[0]

        pred_lat, pred_lng = self.partitionings[-1].get_lat_lng(pred_class)

        return {
            'lat': pred_lat,
            'lng': pred_lng,
            'class': pred_class
        }
