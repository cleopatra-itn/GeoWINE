import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union, List
import torch
import torchvision
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Union
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision
import torch
from argparse import Namespace, ArgumentParser
from datetime import datetime
import json
import logging
from pathlib import Path
import torch
import torchvision
import pytorch_lightning as pl
import utils_global
from s2_utils import Partitioning, Hierarchy

def check_is_valid_torchvision_architecture(architecture: str):
    """Raises an ValueError if architecture is not part of available torchvision models
    """
    available = sorted(
        name
        for name in torchvision.models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torchvision.models.__dict__[name])
    )
    if architecture not in available:
        raise ValueError(f"{architecture} not in {available}")


def build_base_model(arch: str):

    model = torchvision.models.__dict__[arch](pretrained=True)

    # get input dimension before classification layer
    if arch in ["mobilenet_v2"]:
        nfeatures = model.classifier[-1].in_features
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif arch in ["densenet121", "densenet161", "densenet169"]:
        nfeatures = model.classifier.in_features
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif "resne" in arch:
        # usually all ResNet variants
        nfeatures = model.fc.in_features
        model = torch.nn.Sequential(*list(model.children())[:-2])
    else:
        raise NotImplementedError

    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.flatten = torch.nn.Flatten(start_dim=1)
    return model, nfeatures


def load_weights_if_available(
    model: torch.nn.Module, classifier: torch.nn.Module, weights_path: Union[str, Path]
):

    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_classifier = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("classifier"):
            state_dict_classifier[k.replace("classifier.", "")] = w
        else:
            logging.warning(f"Unexpected prefix in state_dict: {k}")
    model.load_state_dict(state_dict_features, strict=True)
    return model, classifier


def vectorized_gc_distance(latitudes, longitudes, latitudes_gt, longitudes_gt):
    R = 6371
    factor_rad = 0.01745329252
    longitudes = factor_rad * longitudes
    longitudes_gt = factor_rad * longitudes_gt
    latitudes = factor_rad * latitudes
    latitudes_gt = factor_rad * latitudes_gt
    delta_long = longitudes_gt - longitudes
    delta_lat = latitudes_gt - latitudes
    subterm0 = torch.sin(delta_lat / 2) ** 2
    subterm1 = torch.cos(latitudes) * torch.cos(latitudes_gt)
    subterm2 = torch.sin(delta_long / 2) ** 2
    subterm1 = subterm1 * subterm2
    a = subterm0 + subterm1
    c = 2 * torch.asin(torch.sqrt(a))
    gcd = R * c
    return gcd


def gcd_threshold_eval(gc_dists, thresholds=[1, 25, 200, 750, 2500]):
    # calculate accuracy for given gcd thresolds
    results = {}
    for thres in thresholds:
        results[thres] = torch.true_divide(
            torch.sum(gc_dists <= thres), len(gc_dists)
        ).item()
    return results


def accuracy(output, target, partitioning_shortnames: list, topk=(1, 5, 10)):
    def _accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = {}
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res[k] = correct_k / batch_size
            return res

    with torch.no_grad():
        out_dict = {}
        for i, pname in enumerate(partitioning_shortnames):
            res_dict = _accuracy(output[i], target[i], topk=topk)
            for k, v in res_dict.items():
                out_dict[f"acc{k}_val/{pname}"] = v

        return out_dict


def summarize_gcd_stats(pnames: List[str], outputs, hierarchy=None):
    gcd_dict = {}
    metric_names = [f"gcd_{p}_val" for p in pnames]
    if hierarchy is not None:
        metric_names.append("gcd_hierarchy_val")
    for metric_name in metric_names:
        distances_flat = [output[metric_name] for output in outputs]
        distances_flat = torch.cat(distances_flat, dim=0)
        gcd_results = gcd_threshold_eval(distances_flat)
        for gcd_thres, acc in gcd_results.items():
            gcd_dict[f"{metric_name}/{gcd_thres}"] = acc
    return gcd_dict


def summarize_test_gcd(pnames, outputs, hierarchy=None):
    def _eval(output):
        # calculate acc@km for a list of given thresholds
        accuracy_outputs = {}
        if hierarchy is not None:
            pnames.append("hierarchy")
        for pname in pnames:
            # concat batches of distances
            distances_flat = torch.cat([x[pname] for x in output], dim=0)
            # acc for all distances
            acc_dict = gcd_threshold_eval(distances_flat)
            accuracy_outputs[f"acc_test/{pname}"] = acc_dict
        return accuracy_outputs

    result = {}

    if isinstance(outputs[0], dict):  # only one testset
        result = _eval(outputs)
    elif isinstance(outputs[0], list):  # multiple testsets
        for testset_index, output in enumerate(outputs):
            result[testset_index] = _eval(output)
    else:
        raise TypeError

    return result


def summarize_loss_acc_stats(pnames: List[str], outputs, topk=[1, 5, 10]):

    loss_acc_dict = {}
    metric_names = []
    for k in topk:
        accuracy_names = [f"acc{k}_val/{p}" for p in pnames]
        metric_names.extend(accuracy_names)
    metric_names.extend([f"loss_val/{p}" for p in pnames])
    for metric_name in ["loss_val/total", *metric_names]:
        metric_total = 0
        for output in outputs:
            metric_value = output[metric_name]
            metric_total += metric_value
        loss_acc_dict[metric_name] = metric_total / len(outputs)
    return loss_acc_dict










class FiveCropImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_csv: Union[str, Path, None],
        image_dir: Union[str, Path],
        img_id_col: Union[str, int] = "img_id",
    ):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        self.image_dir = image_dir
        self.img_id_col = img_id_col
        self.meta_info = None
        if meta_csv is not None:
            print(f"Read {meta_csv}")
            self.meta_info = pd.read_csv(meta_csv)
            self.meta_info["img_path"] = self.meta_info[img_id_col].apply(
                lambda img_id: str(self.image_dir / img_id)
            )
        else:
            image_files = []
            for ext in ["jpg", "jpeg", "png"]:
                image_files.extend([str(p) for p in self.image_dir.glob(f"**/*.{ext}")])
            self.meta_info = pd.DataFrame(image_files, columns=["img_path"])
            self.meta_info[self.img_id_col] = self.meta_info["img_path"].apply(
                lambda x: Path(x).stem
            )
        self.tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        meta = self.meta_info.iloc[idx]
        meta = meta.to_dict()
        meta["img_id"] = meta[self.img_id_col]

        image = Image.open(meta["img_path"]).convert("RGB")
        image = torchvision.transforms.Resize(256)(image)
        crops = torchvision.transforms.FiveCrop(224)(image)
        crops_transformed = []
        for crop in crops:
            crops_transformed.append(self.tfm(crop))
        return torch.stack(crops_transformed, dim=0), meta



class MultiPartitioningClassifier(pl.LightningModule):
    def __init__(self, hparams: Namespace):
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
            partitionings.append(Partitioning(Path(path), shortname, skiprows=2))

        if len(self.hparams.partitionings["files"]) == 1:
            return partitionings, None

        return partitionings, Hierarchy(partitionings)

    def __build_model(self):
        logging.info("Build model")
        model, nfeatures = utils_global.build_base_model(self.hparams.arch)

        classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(nfeatures, len(self.partitionings[i]))
                for i in range(len(self.partitionings))
            ]
        )

        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, classifier = utils_global.load_weights_if_available(
                model, classifier, self.hparams.weights
            )

        return model, classifier

    def forward(self, x):
        fv = self.model(x)
        yhats = [self.classifier[i](fv) for i in range(len(self.partitionings))]
        return yhats

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        images, target = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output = self(images)

        # individual losses per partitioning
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # stats
        losses_stats = {
            f"loss_train/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }
        for metric_name, metric_value in losses_stats.items():
            self.log(metric_name, metric_value, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, **losses_stats}

    def validation_step(self, batch, batch_idx):
        images, target, true_lats, true_lngs = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self(images)

        # loss calculation
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # log top-k accuracy for each partitioning
        individual_accuracy_dict = utils_global.accuracy(
            output, target, [p.shortname for p in self.partitionings]
        )
        # log loss for each partitioning
        individual_loss_dict = {
            f"loss_val/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }

        # log GCD error@km threshold
        distances_dict = {}

        if self.hierarchy is not None:
            hierarchy_logits = [
                yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(output)
            ]
            hierarchy_logits = torch.stack(hierarchy_logits, dim=-1,)
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        pnames = [p.shortname for p in self.partitionings]
        if self.hierarchy is not None:
            pnames.append("hierarchy")
        for i, pname in enumerate(pnames):
            # get predicted coordinates
            if i == len(self.partitionings):
                i = i - 1
                pred_class_indexes = torch.argmax(hierarchy_preds, dim=1)
            else:
                pred_class_indexes = torch.argmax(output[i], dim=1)
            pred_latlngs = [
                self.partitionings[i].get_lat_lng(idx)
                for idx in pred_class_indexes.tolist()
            ]
            pred_lats, pred_lngs = map(list, zip(*pred_latlngs))
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            # calculate error
            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                true_lats.type_as(pred_lats),
                true_lngs.type_as(pred_lats),
            )
            distances_dict[f"gcd_{pname}_val"] = distances

        output = {
            "loss_val/total": loss,
            **individual_accuracy_dict,
            **individual_loss_dict,
            **distances_dict,
        }
        return output

    def validation_epoch_end(self, outputs):
        pnames = [p.shortname for p in self.partitionings]

        # top-k accuracy and loss per partitioning
        loss_acc_dict = utils_global.summarize_loss_acc_stats(pnames, outputs)

        # GCD stats per partitioning
        gcd_dict = utils_global.summarize_gcd_stats(pnames, outputs, self.hierarchy)

        metrics = {
            "val_loss": loss_acc_dict["loss_val/total"],
            **loss_acc_dict,
            **gcd_dict,
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, logger=True)

    def _multi_crop_inference(self, batch):
        images, meta_batch = batch
        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape crop dimension to batch
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        # forward pass
        yhats = self(images)
        yhats = [torch.nn.functional.softmax(yhat, dim=1) for yhat in yhats]

        # respape back to access individual crops
        yhats = [
            torch.reshape(yhat, (cur_batch_size, ncrops, *list(yhat.shape[1:])))
            for yhat in yhats
        ]

        # calculate max over crops
        yhats = [torch.max(yhat, dim=1)[0] for yhat in yhats]

        hierarchy_preds = None
        if self.hierarchy is not None:
            hierarchy_logits = torch.stack(
                [yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)],
                dim=-1,
            )
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        return yhats, meta_batch, hierarchy_preds

    def inference(self, batch):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        pred_class_dict = {}
        pred_lat_dict = {}
        pred_lng_dict = {}
        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            pred_lat_dict[pname] = pred_lats
            pred_lng_dict[pname] = pred_lngs
            pred_class_dict[pname] = pred_classes

        return meta_batch["img_path"], pred_class_dict, pred_lat_dict, pred_lng_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        distances_dict = {}
        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)

            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                meta_batch["latitude"].type_as(pred_lats),
                meta_batch["longitude"].type_as(pred_lngs),
            )
            distances_dict[pname] = distances

        return distances_dict

    def test_epoch_end(self, outputs):
        result = utils_global.summarize_test_gcd(
            [p.shortname for p in self.partitionings], outputs, self.hierarchy
        )
        return {**result}

    def configure_optimizers(self):

        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.optim["params"]
        )

        return {
            "optimizer": optim_feature_extrator,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_feature_extrator, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }





def get_file_name(path):
    head, file_name = os.path.split(path)
    return head, file_name

def get_root_path():
    return Path(os.path.dirname(__file__))

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            d = json.load(json_data)
    except Exception as s:
        d=s
        print(d)
    return d


